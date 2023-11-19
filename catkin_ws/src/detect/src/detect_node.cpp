#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <onnxruntime_cxx_api.h>
#include <std_msgs/Float32MultiArray.h>

template <typename T>
static void softmax(T& input) {
    float rowmax = *std::max_element(input.begin(), input.end());
    std::vector<float> y(input.size(), 0.0f);
    float sum = 0.0f;
    for (size_t i = 0; i != input.size(); ++i) {
        sum += y[i] = std::exp(input[i] - rowmax);
    }
    for (size_t i = 0; i != input.size(); ++i) {
        input[i] = y[i] / sum;
    }
}

class MNIST {
public:
    MNIST(int argc, char** argv) : env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeROS"),
        session(env, "/home/cy/catkin_ws/src/detect/model/mnist (1).onnx", Ort::SessionOptions{nullptr}),
        input_tensor_(nullptr),
        output_tensor_(nullptr),
        it_(nh_) {
        ROS_INFO("Model load success");

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // Create Ort::Value objects for input and output tensors
        input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(),
            std::vector<int64_t>{1, 1, width_, height_}.data(), 4);

        output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(),
            std::vector<int64_t>{1, 10}.data(), 2);

    }

    std::ptrdiff_t Run() {
        const char* input_names[] = { "Input3" };
        const char* output_names[] = { "Plus214_Output_0" };

        Ort::RunOptions run_options;
    
        // Run inference
        session.Run(run_options, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);

        // Perform softmax and get the result
        softmax(results_);
        result_ = static_cast<int>(std::distance(results_.begin(), std::max_element(results_.begin(), results_.end())));

        for (size_t i = 0; i < results_.size(); ++i) {
            ROS_INFO("Probability for digit %zu: %f", i, results_[i]);
        }
        return result_;
        
    }

    void ProcessImageCallback(const sensor_msgs::ImageConstPtr& msg) {
        try {
            ROS_INFO("Image received and processing...");
            cv::Mat img = cv_bridge::toCvShare(msg, "bgr8")->image;
            cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
            cv::resize(img, img, cv::Size(width_, height_));


            // Copy image data to MNIST input array
            for (int i = 0; i < height_; ++i) {
                for (int j = 0; j < width_; ++j) {
                    input_image_[i * width_ + j] = img.at<uchar>(i, j) / 255.0; // Normalize to [0, 1]
                }
            }

            // Run MNIST model inference
            std::ptrdiff_t result = Run();

            // Output the inference result
            ROS_INFO("Forecast figure: %ld", result);

            // Create image message object
            sensor_msgs::ImagePtr output_image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();

            // Publish the image message
            image_pub_.publish(output_image_msg);

        }
        catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }

private:
    Ort::Env env;
    Ort::Session session;
    Ort::Value input_tensor_;
    Ort::Value output_tensor_;

    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_ = it_.subscribe("/usb_cam/image_raw", 1, &MNIST::ProcessImageCallback, this);
    image_transport::Publisher image_pub_ = it_.advertise("/detect_node/output_image", 1);
    static constexpr const int width_ = 28;
    static constexpr const int height_ = 28;
    std::array<float, width_ * height_> input_image_{};
    std::array<float, 10> results_{};
    int result_{0};
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "detect_node");

    MNIST mnist(argc,argv);

    while (ros::ok()) {
        ros::spinOnce();
    }


    return 0;
}
