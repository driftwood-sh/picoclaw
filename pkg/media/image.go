package media

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"image"
	_ "image/gif"  // register GIF for image.DecodeConfig
	_ "image/jpeg" // register JPEG for image.DecodeConfig
	_ "image/png"  // register PNG for image.DecodeConfig
	"io"
	"os"
)

// ErrNotRasterImage is returned when bytes are not a PNG, JPEG, GIF or WebP image.
var ErrNotRasterImage = errors.New("not a PNG, JPEG, GIF or WebP image")

// DetectImageMIME reports the real MIME type of an image read from r.
//
// Vision providers accept only PNG, JPEG, GIF and WebP, and answer HTTP 400
// ("image data you provided does not represent a valid image") for anything
// else, such as an SVG or bytes that are not an image at all. The type a tool
// or channel claimed for a file is not evidence of that, so callers that send
// an image to a provider decide from the bytes. PNG, JPEG and GIF headers are
// parsed; WebP has no decoder in the standard library, so it is recognized by
// its RIFF/WEBP container header.
func DetectImageMIME(r io.Reader) (string, error) {
	br := bufio.NewReader(r)
	// Peek returns what it could read together with the error; a short or
	// empty file simply fails the checks below.
	head, _ := br.Peek(16)
	if isWebP(head) {
		return "image/webp", nil
	}

	_, format, err := image.DecodeConfig(br)
	if err != nil {
		return "", fmt.Errorf("%w: %v", ErrNotRasterImage, err)
	}
	switch format {
	case "png", "jpeg", "gif":
		return "image/" + format, nil
	default:
		return "", fmt.Errorf("%w: %s", ErrNotRasterImage, format)
	}
}

// DetectImageFileMIME is DetectImageMIME for the file at path.
func DetectImageFileMIME(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	return DetectImageMIME(f)
}

// isWebP reports whether head starts a WebP file: a RIFF container whose form
// type is WEBP and whose first chunk is lossy (VP8), lossless (VP8L) or
// extended (VP8X).
func isWebP(head []byte) bool {
	if len(head) < 16 {
		return false
	}
	if !bytes.Equal(head[0:4], []byte("RIFF")) || !bytes.Equal(head[8:12], []byte("WEBP")) {
		return false
	}
	switch string(head[12:16]) {
	case "VP8 ", "VP8L", "VP8X":
		return true
	default:
		return false
	}
}
