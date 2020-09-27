# gst-face-swap

WIP

чтобы плагин был доступен в gst-launch и тд

```bash
export GST_PLUGIN_PATH="/home/rustam/Рабочий стол/face-swap/gst-face-swap/target/debug"
export GST_PLUGIN_PATH="/home/rustam/Рабочий стол/face-swap/gst-face-swap/target/release"
```

тестовый пайплайн

```
gst-launch-1.0 autovideosrc ! videoconvert ! rsfaceswap face=../bradley.jpeg ! videoconvert ! autovideosink
```

пайплайн с транскодингом

```
gst-launch-1.0 filesrc location=../me3.mp4 ! decodebin ! videoconvert ! rsfaceswap ! videoconvert ! nvh264enc ! mpegtsmux ! filesink location=../test_me3.mp4
```

Пайплайн для транскодинга со звуком

```
gst-launch-1.0 -v filesrc location="../hb.mp4" ! decodebin name=dbin ! \
videoconvert ! rsfaceswap ! videoconvert ! nvh264enc ! mpegtsmux name=mux ! \
filesink location=../hb_triangles.mp4 dbin. ! \
audioconvert ! audioresample ! audio/x-raw,rate=48000 ! \
fdkaacenc bitrate=96000 ! audio/mpeg ! aacparse ! audio/mpeg, mpegversion=4 ! mux.
```
