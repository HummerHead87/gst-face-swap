use glib;
use glib::subclass;
use glib::subclass::prelude::*;

use gst;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::subclass::prelude::*;
use gst_base::{self, BaseTransform};
use gst_video;

use std::i32;
use std::sync::Mutex;

use once_cell::sync::Lazy;

use opencv::prelude::*;
use opencv::{
    core::{self, Mat, Point, Scalar, Vector},
    face, imgproc,
    objdetect::CascadeClassifier,
    types,
};

use crate::convert::{cv_mat_to_gst_buf, gst_buffer_to_cv_mat};

static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "rsfaceswap",
        gst::DebugColorFlags::empty(),
        Some("Rust OpenCV face swap plugin"),
    )
});

struct State {
    in_info: gst_video::VideoInfo,
    out_info: gst_video::VideoInfo,
}

// type LandmarkDetector = core::Ptr<dyn face::Facemark>;
// struct LandmarkDetector {
//     landmark_detector: core::Ptr<dyn face::Facemark>,
// }
struct LandmarkDetector(core::Ptr<dyn face::Facemark>);
unsafe impl Send for LandmarkDetector {}
unsafe impl Sync for LandmarkDetector {}

struct FaceSwap {
    state: Mutex<Option<State>>,
    detector: Mutex<CascadeClassifier>,
    landmark_detector: Mutex<LandmarkDetector>,
}

impl FaceSwap {}

impl ObjectSubclass for FaceSwap {
    const NAME: &'static str = "RsFaceSwap";
    type ParentType = gst_base::BaseTransform;
    type Instance = gst::subclass::ElementInstanceStruct<Self>;
    type Class = subclass::simple::ClassStruct<Self>;

    // This macro provides some boilerplate
    glib_object_subclass!();

    fn new() -> Self {
        let detector = CascadeClassifier::new("../haarcascade_frontalface_alt2.xml").unwrap();
        let mut landmark_detector = face::create_facemark_lbf().unwrap();
        landmark_detector.load_model("../lbfmodel.yaml").unwrap();

        Self {
            state: Mutex::new(None),
            detector: Mutex::new(detector),
            landmark_detector: Mutex::new(LandmarkDetector(landmark_detector)),
        }
    }

    fn class_init(klass: &mut subclass::simple::ClassStruct<Self>) {
        klass.set_metadata(
            "Face Swap plugin",
            "Filter/Effect/Converter/Video",
            "Swaping faces, detected on video to face from source",
            "Valeyev Rustam <snooks87@gmail.com>",
        );

        klass.configure(
            gst_base::subclass::BaseTransformMode::NeverInPlace,
            false,
            false,
        );

        let caps = gst::Caps::new_simple(
            "video/x-raw",
            &[
                ("format", &gst_video::VideoFormat::Bgrx.to_str()),
                ("width", &gst::IntRange::<i32>::new(0, i32::MAX)),
                ("height", &gst::IntRange::<i32>::new(0, i32::MAX)),
                (
                    "framerate",
                    &gst::FractionRange::new(
                        gst::Fraction::new(0, 1),
                        gst::Fraction::new(i32::MAX, 1),
                    ),
                ),
            ],
        );

        let src_pad_template = gst::PadTemplate::new(
            "src",
            gst::PadDirection::Src,
            gst::PadPresence::Always,
            &caps,
        )
        .unwrap();
        klass.add_pad_template(src_pad_template);

        let caps = gst::Caps::new_simple(
            "video/x-raw",
            &[
                ("format", &gst_video::VideoFormat::Bgrx.to_str()),
                ("width", &gst::IntRange::<i32>::new(0, i32::MAX)),
                ("height", &gst::IntRange::<i32>::new(0, i32::MAX)),
                (
                    "framerate",
                    &gst::FractionRange::new(
                        gst::Fraction::new(0, 1),
                        gst::Fraction::new(i32::MAX, 1),
                    ),
                ),
            ],
        );
        let sink_pad_template = gst::PadTemplate::new(
            "sink",
            gst::PadDirection::Sink,
            gst::PadPresence::Always,
            &caps,
        )
        .unwrap();
        klass.add_pad_template(sink_pad_template);
    }
}

impl ObjectImpl for FaceSwap {
    glib_object_impl!();
}

impl ElementImpl for FaceSwap {}

impl BaseTransformImpl for FaceSwap {
    fn set_caps(
        &self,
        element: &BaseTransform,
        incaps: &gst::Caps,
        outcaps: &gst::Caps,
    ) -> Result<(), gst::LoggableError> {
        let in_info = match gst_video::VideoInfo::from_caps(incaps) {
            Ok(info) => info,
            Err(_) => return Err(gst_loggable_error!(CAT, "Failed to parse input caps")),
        };
        let out_info = match gst_video::VideoInfo::from_caps(outcaps) {
            Ok(info) => info,
            Err(_) => return Err(gst_loggable_error!(CAT, "Failed to parse output caps")),
        };

        gst_debug!(
            CAT,
            obj: element,
            "Configured for caps {} to {}",
            incaps,
            outcaps,
        );

        *self.state.lock().unwrap() = Some(State { in_info, out_info });

        Ok(())
    }

    fn stop(&self, element: &gst_base::BaseTransform) -> Result<(), gst::ErrorMessage> {
        // Drop state
        let _ = self.state.lock().unwrap().take();

        gst_info!(CAT, obj: element, "Stopped");

        Ok(())
    }

    fn get_unit_size(&self, element: &BaseTransform, caps: &gst::Caps) -> Option<usize> {
        match gst_video::VideoInfo::from_caps(caps).map(|info| info.size()) {
            Ok(size) => Some(size),
            Err(error) => {
                gst_error!(CAT, obj: element, "Failed to parse caps. Error: {}", error);
                None
            }
        }
    }

    fn transform_caps(
        &self,
        element: &BaseTransform,
        direction: gst::PadDirection,
        caps: &gst::Caps,
        filter: Option<&gst::Caps>,
    ) -> Option<gst::Caps> {
        let mut other_caps = caps.clone();

        for s in other_caps.make_mut().iter_mut() {
            s.set("format", &gst_video::VideoFormat::Bgrx.to_str());
        }

        gst_debug!(
            CAT,
            obj: element,
            "Transformed caps from {} to {} in direction {:?}",
            caps,
            other_caps,
            direction
        );

        if let Some(filter) = filter {
            Some(filter.intersect_with_mode(&other_caps, gst::CapsIntersectMode::First))
        } else {
            Some(other_caps)
        }
    }

    fn transform(
        &self,
        element: &BaseTransform,
        inbuf: &gst::Buffer,
        outbuf: &mut gst::BufferRef,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let mut state_guard = self.state.lock().unwrap();
        let state = state_guard.as_mut().ok_or_else(|| {
            gst_element_error!(element, gst::CoreError::Negotiation, ["Have no state yet"]);
            gst::FlowError::NotNegotiated
        })?;
        let width = state.in_info.width();
        let height = state.in_info.height();

        let in_frame =
            gst_video::VideoFrameRef::from_buffer_ref_readable(inbuf.as_ref(), &state.in_info)
                .or_else(|_| {
                    gst_element_error!(
                        element,
                        gst::CoreError::Failed,
                        ["Failed to map input buffer readable"]
                    );
                    Err(gst::FlowError::Error)
                })?;

        let mut frame =
            gst_buffer_to_cv_mat(&in_frame, width as i32, height as i32).or_else(|err| {
                gst_element_error!(
                    element,
                    gst::CoreError::Failed,
                    ["Error convert gst buffer to OpenCV mat: {}", err]
                );
                Err(gst::FlowError::Error)
            })?;

        let mut gray = core::Mat::default().unwrap();
        imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0).unwrap();

        let faces = {
            let mut detector = self.detector.lock().unwrap();
            let mut faces = types::VectorOfRect::new();

            detector
                .detect_multi_scale(
                    &gray,
                    &mut faces,
                    1.05,
                    5,
                    0,
                    core::Size::new(200, 200),
                    core::Size::new(5000, 5000),
                )
                .unwrap();

            faces
        };

        {
            let faces = &faces;
            for face in faces {
                imgproc::rectangle(
                    &mut frame,
                    face,
                    Scalar::new(255.0, 255.0, 255.0, 0.0),
                    2,
                    1,
                    0,
                )
                .unwrap();
            }
        }

        let landmarks = {
            let mut landmarks: Vector<Vector<core::Point2f>> = Vector::new();

            let mut landmark_detector = self.landmark_detector.lock().unwrap();
            landmark_detector
                .0
                .fit(&gray, &faces, &mut landmarks)
                .unwrap();

            landmarks
        };

        for landmark in landmarks {
            // print!("{:?}", landmark);
            // let mut points: Vector<Point> = Vector::new();
            // print!("landmarks count: {}", landmark.len());
            // let landmark_clone = landmark.clone();
            // let landmark_vec = landmark.to_vec();

            // {
            //     let landmark = &landmark;
            //     for point in landmark {
            //         let x = point.x;
            //         let y = point.y;

            //         let point = Point::new(x as i32, y as i32);
            //         // points.push(point);

            //         imgproc::circle(
            //             &mut frame,
            //             point,
            //             2,
            //             Scalar::new(0.0, 0.0, 255.0, 0.0),
            //             -1,
            //             1,
            //             0,
            //         )
            //         .unwrap();
            //     }
            // }

            let mut convexhull = types::VectorOfPoint::new();
            let points = {
                let landmark = &landmark;
                types::VectorOfPoint::from_iter(landmark.iter().map(|p| p.to().unwrap()))
            };

            imgproc::convex_hull(&points, &mut convexhull, true, true).unwrap();
            // imgproc::polylines(
            //     &mut frame,
            //     &convexhull,
            //     true,
            //     Scalar::new(255.0, 0.0, 0.0, 0.0),
            //     3,
            //     1,
            //     0,
            // )
            // .unwrap();

            // Delaunau triangulation
            let rect = imgproc::bounding_rect(&convexhull).unwrap();
            let mut subdiv = imgproc::Subdiv2D::new(rect).unwrap();
            subdiv.insert_multiple(&landmark).unwrap();

            let mut triangles = types::VectorOfVec6f::new();
            subdiv.get_triangle_list(&mut triangles).unwrap();

            for t in triangles {
                let pt1 = Point::new(t[0] as i32, t[1] as i32);
                let pt2 = Point::new(t[2] as i32, t[3] as i32);
                let pt3 = Point::new(t[4] as i32, t[5] as i32);

                imgproc::line(
                    &mut frame,
                    pt1,
                    pt2,
                    Scalar::new(0.0, 0.0, 255.0, 0.0),
                    2,
                    1,
                    0,
                )
                .unwrap();
                imgproc::line(
                    &mut frame,
                    pt2,
                    pt3,
                    Scalar::new(0.0, 0.0, 255.0, 0.0),
                    2,
                    1,
                    0,
                )
                .unwrap();
                imgproc::line(
                    &mut frame,
                    pt3,
                    pt1,
                    Scalar::new(0.0, 0.0, 255.0, 0.0),
                    2,
                    1,
                    0,
                )
                .unwrap();
            }
        }

        let mut out_frame =
            gst_video::VideoFrameRef::from_buffer_ref_writable(outbuf, &state.out_info).or_else(
                |_| {
                    gst_element_error!(
                        element,
                        gst::CoreError::Failed,
                        ["Failed to map output buffer writable"]
                    );
                    Err(gst::FlowError::Error)
                },
            )?;

        cv_mat_to_gst_buf(&mut out_frame, &frame, width as usize).or_else(|err| {
            gst_element_error!(
                element,
                gst::CoreError::Failed,
                ["Cant write to gst buffer from OpenCV mat: {}", err]
            );
            Err(gst::FlowError::Error)
        })?;

        Ok(gst::FlowSuccess::Ok)
    }
}

pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "rsfaceswap",
        gst::Rank::None,
        FaceSwap::get_type(),
    )
}

fn find_index_in_vec(list: &Vec<core::Point2f>, x: f32, y: f32) -> Option<usize> {
    list.iter().position(|&pt| pt.x == x && pt.y == y)
}
