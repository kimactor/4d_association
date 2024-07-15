#include <vector>
#include <opencv2/opencv.hpp>

#include "../src/pose.hpp"
#include "../src/poseEstimation.hpp"
#include <time.h>

#define TEST_VIDEO
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
using namespace Ort;

int printModelInfo(const std::string& model_path)
{
	std::cout << " model_path:  " << model_path << std::endl;
	Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "lightweight_openpose");

	Ort::SessionOptions session_options;
	//OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0); //tensorRT
	auto status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	Session* session = new Session(env, std::wstring(model_path.begin(), model_path.end()).c_str(), session_options);
	std::vector<const char*> input_names = { "data" };
	Ort::AllocatorWithDefaultOptions allocator;

	//session->GetInputTypeInfo(input_names);
	size_t num_input_nodes = session->GetInputCount();
	size_t num_output_nodes = session->GetOutputCount();

	std::cout << "Number of input node is:" << num_input_nodes << std::endl;
	std::cout << "Number of output node is:" << num_output_nodes << std::endl;
	//获取输入输出维度
	for (auto i = 0; i < num_input_nodes; i++)
	{
		std::vector<int64_t> input_dims = session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		std::cout << std::endl << "input " << i << " dim is: ";
		for (auto j = 0; j < input_dims.size(); j++)
			std::cout << input_dims[j] << " ";
	}
	for (auto i = 0; i < num_output_nodes; i++)
	{
		std::vector<int64_t> output_dims = session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		std::cout << std::endl << "output " << i << " dim is: ";
		for (auto j = 0; j < output_dims.size(); j++)
			std::cout << output_dims[j] << " ";
	}
	//输入输出的节点名
	std::cout << std::endl;//换行输出
	for (auto i = 0; i < num_input_nodes; i++)
		std::cout << "The input op-name " << i << " is:" << session->GetInputNameAllocated(i, allocator) << std::endl;
	for (auto i = 0; i < num_output_nodes; i++)
		std::cout << "The output op-name " << i << " is:" << session->GetOutputNameAllocated(i, allocator) << std::endl;


	std::cout << std::endl;
	return 0;
}


int main(){

	const std::string model_path = "../data/models/poseEstimationModel.onnx";
	printModelInfo(model_path);

	const std::string model_path_1 = "D:/work/Find_demo/lightweight-human-pose-estimation.pytorch/data/poseEstimationModel_1.onnx";
	printModelInfo(model_path_1);
	//return 0;


	poseEstimation::poseEstimation pe(model_path_1);
	poseEstimation::poseTracker pt;

#ifdef TEST_VIDEO
	//const std::string video_path = "../data/save_video.mp4";
	//cv::VideoCapture cap(video_path);
	cv::VideoCapture cap(0, cv::CAP_ANY);

	cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
	cap.set(cv::CAP_PROP_FPS, 60);
	while (true)
	{
		if (!cap.isOpened())
			break;
		cv::Mat frame;
		clock_t pose_time = clock();
		cap >> frame;
		std::vector<poseEstimation::Pose> poses = pe.run(frame);
		pt.track(poses);
		for (int i = 0; i < poses.size(); i++)
			poses[i].draw(frame, true);
		cv::imshow("lightweight openpose", frame);

		if (cv::waitKey(1) == 'q')
			break;
		std::cout << " pose use time: " << clock() - pose_time << std::endl;
	}
#else
	cv::Mat img = cv::imread("../data/input.jpg");
	std::vector<poseEstimation::Pose> poses = pe.run(img);
	pt.track(poses);
	for(int i = 0; i < poses.size(); i++)
		poses[i].draw(img, true);
	cv::imwrite("../data/output.jpg", img);
#endif
	return 0;
}
