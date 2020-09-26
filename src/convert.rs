extern crate opencv;
use anyhow::{anyhow, Error};
use gst;
use gst_base::BaseTransform;
use gst_video::{VideoFormat, VideoFrameRef};
use gstreamer::buffer::BufferRef;
use opencv::core::{self, Mat};
use opencv::prelude::*;

pub fn gst_buffer_to_cv_mat(
    frame: &VideoFrameRef<&BufferRef>,
    width: i32,
    height: i32,
) -> Result<Mat, Error> {
    // неправильно работает, т.к. opencv считает что pixel stride равен 3, а у gstreamer прилетает
    // изображение с pixel stride 4
    // let frame: Mat;
    // let size = core::Size::new(width as i32, height as i32);

    // unsafe {
    //     let ptr = in_data.as_ptr() as *mut std::ffi::c_void;

    //     frame = Mat::new_size_with_data(size, core::CV_8UC3, ptr, in_stride).unwrap();
    // }

    // рабочая версия с лишней аллокацией и перебором всех пикселей изображения
    let mut mat = Mat::zeros(height, width, core::CV_8UC3)
        .unwrap()
        .to_mat()
        .unwrap();

    let stride = frame.plane_stride()[0] as usize;
    let data = frame.plane_data(0).unwrap();
    let line_bytes = (width * 4) as usize;

    for (row, line) in data.chunks_exact(stride).enumerate() {
        for (col, in_p) in line[..line_bytes].chunks_exact(4).enumerate() {
            assert_eq!(in_p.len(), 4);
            let mut px: &mut core::Vec3b = mat
                .at_2d_mut(row as i32, col as i32)
                .or_else(|err| Err(anyhow!("Can't get pixel from map: {}", err)))?;

            px.0[0] = in_p[0];
            px.0[1] = in_p[1];
            px.0[2] = in_p[2];
        }
    }

    Ok(mat)
}

pub fn cv_mat_to_gst_buf(
    frame: &mut VideoFrameRef<&mut BufferRef>,
    mat: &Mat,
    width: usize,
) -> Result<(), Error> {
    let out_stride = frame.plane_stride()[0] as usize;
    let out_format = frame.format();
    let out_data = frame.plane_data_mut(0).unwrap();
    let out_line_bytes = width * 4;

    if out_format == VideoFormat::Bgrx {
        assert_eq!(mat.rows(), (out_data.len() / out_stride) as i32);
        for (row, out_line) in out_data.chunks_exact_mut(out_stride).enumerate() {
            for (col, out_p) in out_line[..out_line_bytes].chunks_exact_mut(4).enumerate() {
                assert_eq!(out_p.len(), 4);
                let px: &core::Vec3b = mat
                    .at_2d(row as i32, col as i32)
                    .or_else(|err| Err(anyhow!("Cant get pixel from map: {}", err)))?;

                out_p[0] = px.0[0];
                out_p[1] = px.0[1];
                out_p[2] = px.0[2];
            }
        }
    } else {
        unimplemented!();
    }

    Ok(())
}
