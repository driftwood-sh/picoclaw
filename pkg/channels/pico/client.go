package pico

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/channels"
	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/identity"
	"github.com/sipeed/picoclaw/pkg/logger"
)

// Reconnect tuning. The server may close the socket on purpose at any time
// (the driftwood relay recycles every connection after a short, jittered
// lifetime), so a dropped connection is routine, not an error.
const (
	// reconnectBaseBackoff is the first wait after a failed dial; it doubles
	// per consecutive failure up to reconnectMaxBackoff. Each wait is jittered
	// to [d/2, d] so a fleet of gateways does not redial in lockstep.
	reconnectBaseBackoff = 500 * time.Millisecond
	reconnectMaxBackoff  = 15 * time.Second
	// reconnectPollInterval is a safety net only: a dropped connection wakes
	// reconnectLoop at once through the redial channel.
	reconnectPollInterval = 1 * time.Second
	// Every redial first waits a jittered redialMinDelay..redialMaxDelay, so
	// a flapping connection can never become a hot loop.
	redialMinDelay = 250 * time.Millisecond
	redialMaxDelay = 750 * time.Millisecond
	// A connection that lived less than this counts as a failed dial for
	// backoff. Example: two gateway processes with one token, where the
	// server's register closes the other's socket on every dial.
	minStableConnection = 5 * time.Second
)

// sendReconnectWait is how long Send parks on a dropped connection, waiting
// for reconnectLoop to install a new one, before it gives up with
// ErrSendFailed. Without it, a reply produced during the sub-second gap of a
// routine server-side recycle was dropped for good (the channel manager
// treats ErrSendFailed as permanent). A var so tests can shorten it.
var sendReconnectWait = 30 * time.Second

// PicoClientChannel connects to a remote Pico Protocol WebSocket server.
type PicoClientChannel struct {
	*channels.BaseChannel
	config *config.PicoClientSettings
	conn   *picoConn
	mu     sync.Mutex
	ctx    context.Context
	cancel context.CancelFunc
	// connReady is closed and replaced (under mu) each time dial installs a
	// new connection, waking every Send parked in liveConn.
	connReady chan struct{}
	// redial wakes reconnectLoop the moment a connection's read loop exits.
	redial chan struct{}
}

// NewPicoClientChannel creates a new Pico Protocol client channel.
func NewPicoClientChannel(
	bc *config.Channel,
	cfg *config.PicoClientSettings,
	messageBus *bus.MessageBus,
) (*PicoClientChannel, error) {
	if cfg.URL == "" {
		return nil, fmt.Errorf("pico_client url is required")
	}

	base := channels.NewBaseChannel("pico_client", cfg, messageBus, bc.AllowFrom)

	return &PicoClientChannel{
		BaseChannel: base,
		config:      cfg,
		connReady:   make(chan struct{}),
		redial:      make(chan struct{}, 1),
	}, nil
}

// Start dials the remote server and begins reading.
func (c *PicoClientChannel) Start(ctx context.Context) error {
	logger.InfoC("pico_client", "Starting Pico Client channel")
	c.ctx, c.cancel = context.WithCancel(ctx)

	if err := c.dial(); err != nil {
		c.cancel()
		return fmt.Errorf("pico_client initial connect: %w", err)
	}

	c.SetRunning(true)
	go c.reconnectLoop()

	logger.InfoCF("pico_client", "Connected", map[string]any{"url": c.config.URL})
	return nil
}

// Stop closes the connection.
func (c *PicoClientChannel) Stop(ctx context.Context) error {
	logger.InfoC("pico_client", "Stopping Pico Client channel")
	c.SetRunning(false)
	if c.cancel != nil {
		c.cancel()
	}
	c.mu.Lock()
	if c.conn != nil {
		c.conn.close()
	}
	c.mu.Unlock()
	logger.InfoC("pico_client", "Pico Client channel stopped")
	return nil
}

func (c *PicoClientChannel) dial() error {
	header := http.Header{}
	if c.config.Token.String() != "" {
		header.Set("Authorization", "Bearer "+c.config.Token.String())
	}

	ws, resp, err := websocket.DefaultDialer.DialContext(c.ctx, c.config.URL, header)
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		return err
	}

	connCtx, connCancel := context.WithCancel(c.ctx)

	pc := &picoConn{
		id:        uuid.New().String(),
		conn:      ws,
		sessionID: c.config.SessionID,
		cancel:    connCancel,
	}
	if pc.sessionID == "" {
		pc.sessionID = uuid.New().String()
	}

	c.mu.Lock()
	c.conn = pc
	close(c.connReady)
	c.connReady = make(chan struct{})
	c.mu.Unlock()

	go c.readLoop(connCtx, pc)
	return nil
}

// signalRedial wakes reconnectLoop without blocking (one pending wake is
// enough; the loop re-reads the connection state itself).
func (c *PicoClientChannel) signalRedial() {
	select {
	case c.redial <- struct{}{}:
	default:
	}
}

// reconnectBackoff returns the jittered wait after the n-th consecutive
// failed dial (n >= 1): uniform in [d/2, d] with d = base * 2^(n-1), capped.
func reconnectBackoff(n int) time.Duration {
	d := reconnectMaxBackoff
	if n < 16 {
		d = min(reconnectBaseBackoff<<(n-1), reconnectMaxBackoff)
	}
	half := d / 2
	return half + rand.N(half+1)
}

// redialDelay is the jittered wait before any redial.
func redialDelay() time.Duration {
	return redialMinDelay + rand.N(redialMaxDelay-redialMinDelay+1)
}

// liveConn returns the current open connection, parking up to
// sendReconnectWait for reconnectLoop to install one if it is down.
func (c *PicoClientChannel) liveConn(ctx context.Context) (*picoConn, error) {
	deadline := time.NewTimer(sendReconnectWait)
	defer deadline.Stop()
	for {
		c.mu.Lock()
		pc, ready := c.conn, c.connReady
		c.mu.Unlock()
		if pc != nil && !pc.closed.Load() {
			return pc, nil
		}
		select {
		case <-ready:
		case <-deadline.C:
			return nil, channels.ErrSendFailed
		case <-ctx.Done():
			return nil, channels.ErrSendFailed
		case <-c.ctx.Done():
			return nil, channels.ErrNotRunning
		}
	}
}

// reconnectLoop re-dials soon after the connection drops (a short jittered
// delay), with jittered exponential backoff while dials keep failing or
// connections keep dying young.
func (c *PicoClientChannel) reconnectLoop() {
	failures := 0
	// Start dialed just before this loop runs.
	connectedAt := time.Now()
	for {
		select {
		case <-c.ctx.Done():
			return
		default:
		}

		c.mu.Lock()
		pc := c.conn
		c.mu.Unlock()

		if pc == nil || pc.closed.Load() {
			if !connectedAt.IsZero() {
				// Judge the connection that just ended exactly once.
				if time.Since(connectedAt) < minStableConnection {
					failures++
				} else {
					failures = 0
				}
				connectedAt = time.Time{}
			}
			wait := redialDelay()
			if failures > 0 {
				wait = max(wait, reconnectBackoff(failures))
			}
			select {
			case <-c.ctx.Done():
				return
			case <-time.After(wait):
			}
			logger.InfoC("pico_client", "Reconnecting...")
			if err := c.dial(); err != nil {
				failures++
				logger.WarnCF("pico_client", "Reconnect failed", map[string]any{
					"error":   err.Error(),
					"attempt": failures,
				})
				continue
			}
			connectedAt = time.Now()
			logger.InfoC("pico_client", "Reconnected")
		}

		select {
		case <-c.ctx.Done():
			return
		case <-c.redial:
		case <-time.After(reconnectPollInterval):
		}
	}
}

func (c *PicoClientChannel) readLoop(connCtx context.Context, pc *picoConn) {
	defer func() {
		pc.close()
		c.signalRedial()
	}()

	readTimeout := time.Duration(c.config.ReadTimeout) * time.Second
	if readTimeout <= 0 {
		readTimeout = 60 * time.Second
	}

	_ = pc.conn.SetReadDeadline(time.Now().Add(readTimeout))
	pc.conn.SetPongHandler(func(string) error {
		return pc.conn.SetReadDeadline(time.Now().Add(readTimeout))
	})

	pingInterval := time.Duration(c.config.PingInterval) * time.Second
	if pingInterval <= 0 {
		pingInterval = 30 * time.Second
	}
	go c.pingLoop(connCtx, pc, pingInterval)

	for {
		select {
		case <-connCtx.Done():
			return
		default:
		}

		_, raw, err := pc.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(
				err,
				websocket.CloseGoingAway,
				websocket.CloseNormalClosure,
			) {
				logger.DebugCF("pico_client", "Read error", map[string]any{
					"error": err.Error(),
				})
			}
			return
		}

		_ = pc.conn.SetReadDeadline(time.Now().Add(readTimeout))

		var msg PicoMessage
		if err := json.Unmarshal(raw, &msg); err != nil {
			continue
		}

		c.handleInbound(pc, msg)
	}
}

func (c *PicoClientChannel) pingLoop(connCtx context.Context, pc *picoConn, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-connCtx.Done():
			return
		case <-ticker.C:
			if pc.closed.Load() {
				return
			}
			pc.writeMu.Lock()
			err := pc.conn.WriteMessage(websocket.PingMessage, nil)
			pc.writeMu.Unlock()
			if err != nil {
				return
			}
		}
	}
}

// handleInbound processes messages from the remote server.
// In client mode the server sends message.create (responses) and the client
// sends message.send (user input). We treat message.create from the server
// as inbound user messages to feed into the agent loop.
func (c *PicoClientChannel) handleInbound(pc *picoConn, msg PicoMessage) {
	switch msg.Type {
	case TypePong:
		// response to our ping, ignore
	case TypeMessageCreate:
		// Server sent us a message — treat as inbound
		c.handleServerMessage(pc, msg)
	case TypeMediaCreate:
		c.handleServerMessage(pc, msg)
	default:
		logger.DebugCF("pico_client", "Ignoring message type", map[string]any{
			"type": msg.Type,
		})
	}
}

func (c *PicoClientChannel) handleServerMessage(pc *picoConn, msg PicoMessage) {
	if isThoughtPayload(msg.Payload) {
		return
	}

	content, _ := msg.Payload[PayloadKeyContent].(string)
	media, err := parseInlineImageMedia(msg.Payload)
	if err != nil {
		logger.WarnCF("pico_client", "Ignoring invalid media payload", map[string]any{
			"error": err.Error(),
		})
		if strings.TrimSpace(content) == "" {
			return
		}
		media = nil
	}
	if strings.TrimSpace(content) == "" && len(media) == 0 {
		return
	}

	sessionID := msg.SessionID
	if sessionID == "" {
		sessionID = pc.sessionID
	}

	chatID := "pico_client:" + sessionID
	senderID := "pico-remote"
	sender := bus.SenderInfo{
		Platform:    "pico_client",
		PlatformID:  senderID,
		CanonicalID: identity.BuildCanonicalID("pico_client", senderID),
	}

	if !c.IsAllowedSender(sender) {
		return
	}

	inboundCtx := bus.InboundContext{
		Channel:   "pico_client",
		ChatID:    chatID,
		ChatType:  "direct",
		SenderID:  senderID,
		MessageID: msg.ID,
		Raw: map[string]string{
			"platform":   "pico_client",
			"session_id": sessionID,
		},
	}

	c.HandleInboundContext(c.ctx, chatID, content, media, inboundCtx, sender)
}

// Send sends a message to the remote server.
func (c *PicoClientChannel) Send(ctx context.Context, msg bus.OutboundMessage) ([]string, error) {
	if !c.IsRunning() {
		return nil, channels.ErrNotRunning
	}

	outMsg := newMessage(TypeMessageSend, map[string]any{
		PayloadKeyContent: msg.Content,
	})
	outMsg.SessionID = strings.TrimPrefix(msg.ChatID, "pico_client:")

	// A write can only be retried when it provably put nothing on the wire:
	// the connection was already closed locally, or the server's close frame
	// had arrived (gorilla's ErrCloseSent). Any other error is returned as-is
	// for the manager's retry policy, which calls Send again.
	for attempt := 0; ; attempt++ {
		pc, err := c.liveConn(ctx)
		if err != nil {
			return nil, err
		}
		err = pc.writeJSON(outMsg)
		if err == nil || attempt > 0 {
			return nil, err
		}
		if !errors.Is(err, errConnClosed) && !errors.Is(err, websocket.ErrCloseSent) {
			return nil, err
		}
		// The server is recycling this connection; make sure it is torn
		// down so liveConn waits for its replacement instead of reusing it.
		pc.close()
	}
}

// StartTyping implements channels.TypingCapable.
func (c *PicoClientChannel) StartTyping(ctx context.Context, chatID string) (func(), error) {
	c.mu.Lock()
	pc := c.conn
	c.mu.Unlock()
	if pc == nil || pc.closed.Load() {
		return func() {}, nil
	}

	startMsg := newMessage(TypeTypingStart, nil)
	startMsg.SessionID = strings.TrimPrefix(chatID, "pico_client:")
	if err := pc.writeJSON(startMsg); err != nil {
		return func() {}, err
	}
	return func() {
		c.mu.Lock()
		currentPC := c.conn
		c.mu.Unlock()
		if currentPC == nil {
			return
		}
		stopMsg := newMessage(TypeTypingStop, nil)
		stopMsg.SessionID = strings.TrimPrefix(chatID, "pico_client:")
		currentPC.writeJSON(stopMsg)
	}, nil
}
