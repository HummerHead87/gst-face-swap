#[macro_use]
extern crate glib;
#[macro_use]
extern crate gstreamer as gst;
extern crate gstreamer_base as gst_base;
extern crate gstreamer_video as gst_video;
extern crate once_cell;

pub mod convert;
mod faceswap;

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    faceswap::register(plugin)?;
    Ok(())
}

// Static plugin metdata that is directly stored in the plugin shared object and read by GStreamer
// upon loading.
// Plugin name, plugin description, plugin entry point function, version number of this plugin,
// license of the plugin, source package name, binary package name, origin where it comes from
// and the date/time of release.
gst_plugin_define!(
    rstest,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION"), "-", env!("COMMIT_ID")),
    "MIT/X11",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    env!("BUILD_REL_DATE")
);
