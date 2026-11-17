package media

import (
	"bytes"
	"errors"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"testing"
)

func encodeTestImages(t *testing.T) (pngData, jpegData, gifData []byte) {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 3, 2))

	var pngBuf, jpegBuf, gifBuf bytes.Buffer
	if err := png.Encode(&pngBuf, img); err != nil {
		t.Fatal(err)
	}
	if err := jpeg.Encode(&jpegBuf, img, nil); err != nil {
		t.Fatal(err)
	}
	pal := image.NewPaletted(image.Rect(0, 0, 3, 2), color.Palette{color.Black, color.White})
	if err := gif.Encode(&gifBuf, pal, nil); err != nil {
		t.Fatal(err)
	}
	return pngBuf.Bytes(), jpegBuf.Bytes(), gifBuf.Bytes()
}

func TestDetectImageMIME_Accepts(t *testing.T) {
	pngData, jpegData, gifData := encodeTestImages(t)
	webpData := append([]byte("RIFF\x1a\x00\x00\x00WEBPVP8L"), make([]byte, 16)...)

	cases := []struct {
		name string
		data []byte
		want string
	}{
		{"png", pngData, "image/png"},
		{"jpeg", jpegData, "image/jpeg"},
		{"gif", gifData, "image/gif"},
		{"webp", webpData, "image/webp"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := DetectImageMIME(bytes.NewReader(tc.data))
			if err != nil {
				t.Fatalf("DetectImageMIME() error = %v", err)
			}
			if got != tc.want {
				t.Fatalf("DetectImageMIME() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestDetectImageMIME_Rejects(t *testing.T) {
	pngData, _, _ := encodeTestImages(t)

	cases := []struct {
		name string
		data []byte
	}{
		{"empty", nil},
		{"svg", []byte(`<svg xmlns="http://www.w3.org/2000/svg"></svg>`)},
		{"text", []byte("not an image")},
		{"png signature only", pngData[:8]},
		{"png signature and zeros", append(append([]byte{}, pngData[:8]...), make([]byte, 64)...)},
		{"riff but not webp", []byte("RIFF\x1a\x00\x00\x00WAVEfmt " + "\x00\x00\x00\x00")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := DetectImageMIME(bytes.NewReader(tc.data))
			if !errors.Is(err, ErrNotRasterImage) {
				t.Fatalf("DetectImageMIME() = %q, %v; want ErrNotRasterImage", got, err)
			}
		})
	}
}

func TestDetectImageFileMIME(t *testing.T) {
	pngData, _, _ := encodeTestImages(t)
	dir := t.TempDir()

	pngPath := filepath.Join(dir, "shot.png")
	if err := os.WriteFile(pngPath, pngData, 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := DetectImageFileMIME(pngPath)
	if err != nil || got != "image/png" {
		t.Fatalf("DetectImageFileMIME(png) = %q, %v; want image/png", got, err)
	}

	if _, err := DetectImageFileMIME(filepath.Join(dir, "missing.png")); err == nil {
		t.Fatal("expected an error for a missing file")
	}
}
