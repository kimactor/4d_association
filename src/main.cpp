#include "kruskal_associater.h"
#include "skel_updater.h"
#include "skel_painter.h"
#include "openpose.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <json/json.h>
#include <time.h>
#include "poseEstimation.hpp"




int ReadFromSeqDataset()
{
	const std::string dataset = "shelf";
	std::map<std::string, Camera> cameras = ParseCameras("../data/" + dataset + "/calibration.json");
	Eigen::Matrix3Xf projs(3, cameras.size() * 4);
	std::vector<cv::Mat> rawImgs(cameras.size());
	std::vector<cv::VideoCapture> videos(cameras.size());
	std::vector<std::vector<OpenposeDetection>> seqDetections(cameras.size());
	const SkelDef& skelDef = GetSkelDef(SKEL19);
	std::vector<std::map<int, Eigen::Matrix4Xf>> skels;

	//#pragma omp parallel for
	for (int i = 0; i < cameras.size(); i++) {
		auto iter = std::next(cameras.begin(), i);
		videos[i] = cv::VideoCapture("../data/" + dataset + "/video/" + iter->first + ".mp4");
		videos[i].set(cv::CAP_PROP_POS_FRAMES, 0);

		projs.middleCols(4 * i, 4) = iter->second.eiProj;
		seqDetections[i] = ParseDetections("../data/" + dataset + "/detection/" + iter->first + ".txt");
		cv::Size imgSize(int(videos[i].get(cv::CAP_PROP_FRAME_WIDTH)), int(videos[i].get(cv::CAP_PROP_FRAME_HEIGHT)));
		for (auto&& detection : seqDetections[i]) {
			for (auto&& joints : detection.joints) {
				joints.row(0) *= (imgSize.width - 1);
				joints.row(1) *= (imgSize.height - 1);
			}
		}
		rawImgs[i].create(imgSize, CV_8UC3);
	}

	KruskalAssociater associater(SKEL19, cameras);
	associater.SetMaxTempDist(0.3f);
	associater.SetMaxEpiDist(0.15f);
	associater.SetEpiWeight(1.f);
	associater.SetTempWeight(2.f);
	associater.SetViewWeight(1.f);
	associater.SetPafWeight(2.f);
	associater.SetHierWeight(1.f);
	associater.SetViewCntWelsh(1.0);
	associater.SetMinCheckCnt(10);
	associater.SetNodeMultiplex(true);
	associater.SetNormalizeEdge(true);			// new feature

	SkelPainter skelPainter(SKEL19);
	skelPainter.rate = 512.f / float(cameras.begin()->second.imgSize.width);
	SkelFittingUpdater skelUpdater(SKEL19, "../data/skel/SKEL19");
	skelUpdater.SetTemporalTransTerm(1e-1f / std::powf(skelPainter.rate, 2));
	skelUpdater.SetTemporalPoseTerm(1e-1f / std::powf(skelPainter.rate, 2));
	cv::Mat detectImg, assocImg, reprojImg;
	cv::Mat resizeImg;
	for (int frameIdx = 0; ; frameIdx++) {
		bool flag = true;
		for (int view = 0; view < cameras.size(); view++) {
			videos[view] >> rawImgs[view];
			if (rawImgs[view].empty()) {
				flag = false;
				break;
			}
			cv::resize(rawImgs[view], rawImgs[view], cv::Size(), skelPainter.rate, skelPainter.rate);
			associater.SetDetection(view, seqDetections[view][frameIdx].Mapping(SKEL19));
		}
		if (!flag)
			break;

		std::clock_t s_time = clock();
		associater.SetSkels3dPrev(skelUpdater.GetSkel3d());
		associater.Associate();
		skelUpdater.Update(associater.GetSkels2d(), projs);
		std::cout << " associator use time: " << clock() - s_time << std::endl;

		// save
		const int layoutCols = 3;
		std::vector<cv::Rect> rois = SkelPainter::MergeImgs(rawImgs, detectImg, layoutCols,
			{ rawImgs.begin()->cols, rawImgs.begin()->rows });
		detectImg.copyTo(assocImg);
		detectImg.copyTo(reprojImg);

#pragma omp parallel for
		for (int view = 0; view < cameras.size(); view++) {
			const OpenposeDetection detection = seqDetections[view][frameIdx].Mapping(SKEL19);
			skelPainter.DrawDetect(detection.joints, detection.pafs, detectImg(rois[view]));
			for (const auto& skel2d : associater.GetSkels2d())
				skelPainter.DrawAssoc(skel2d.second.middleCols(view * skelDef.jointSize, skelDef.jointSize), assocImg(rois[view]), skel2d.first);

			for (const auto& skel3d : skelUpdater.GetSkel3d())
				skelPainter.DrawReproj(skel3d.second, projs.middleCols(4 * view, 4), reprojImg(rois[view]), skel3d.first);
		}

		skels.emplace_back(skelUpdater.GetSkel3d());
		cv::imshow("reproj", reprojImg);
		cv::waitKey(1);
		std::cout << std::to_string(frameIdx) << std::endl;
	}

	SerializeSkels(skels, "../output/skel.txt");

	return 0;
}

int run_test()
{
	const std::string dataset = "shelf";
	std::map<std::string, Camera> cameras = ParseCameras("../data/" + dataset + "/calibration.json");
	Eigen::Matrix3Xf projs(3, cameras.size() * 4);
	std::vector<cv::Mat> rawImgs(cameras.size());
	std::vector<cv::VideoCapture> videos(cameras.size());
	std::vector<std::map<int, Eigen::Matrix4Xf>> skels;
	const SkelDef& skelDef = GetSkelDef(COCO18);
	poseEstimation::poseEstimation pe("../data/models/poseEstimationModel.onnx");

	for (int i = 0; i < cameras.size(); i++) {
		auto iter = std::next(cameras.begin(), i);
		videos[i] = cv::VideoCapture("../data/" + dataset + "/video/" + iter->first + ".mp4");
		videos[i].set(cv::CAP_PROP_POS_FRAMES, 0);

		projs.middleCols(4 * i, 4) = iter->second.eiProj;
		cv::Size imgSize(int(videos[i].get(cv::CAP_PROP_FRAME_WIDTH)), int(videos[i].get(cv::CAP_PROP_FRAME_HEIGHT)));
		rawImgs[i].create(imgSize, CV_8UC3);
	}

	KruskalAssociater associater(COCO18, cameras);
	associater.SetMaxTempDist(0.3f);
	associater.SetMaxEpiDist(0.15f);
	associater.SetEpiWeight(1.f);
	associater.SetTempWeight(2.f);
	associater.SetViewWeight(1.f);
	associater.SetPafWeight(2.f);
	associater.SetHierWeight(1.f);
	associater.SetViewCntWelsh(1.0);
	associater.SetMinCheckCnt(10);
	associater.SetNodeMultiplex(true);
	associater.SetNormalizeEdge(true);			// new feature

	SkelPainter skelPainter(COCO18);
	skelPainter.rate = 512.f / float(cameras.begin()->second.imgSize.width);
	SkelFittingUpdater skelUpdater(COCO18, "../data/skel/COCO18");
	skelUpdater.SetTemporalTransTerm(1e-1f / std::powf(skelPainter.rate, 2));
	skelUpdater.SetTemporalPoseTerm(1e-1f / std::powf(skelPainter.rate, 2));
	cv::Mat detectImg, assocImg, reprojImg;
	cv::Mat resizeImg;

	for (int frameIdx = 0; ; frameIdx++) {
		std::vector<OpenposeDetection> seqDetections(cameras.size(), OpenposeDetection(COCO18));
		bool flag = true;
		for (int view = 0; view < cameras.size(); view++) {
			videos[view] >> rawImgs[view];
			if (rawImgs[view].empty()) {
				flag = false;
				break;
			}
			pe.run(rawImgs[view], seqDetections[view]);
			cv::resize(rawImgs[view], rawImgs[view], cv::Size(), skelPainter.rate, skelPainter.rate);
			associater.SetDetection(view, seqDetections[view]);

		}
		if (!flag)
			break;

		std::clock_t s_time = clock();
		associater.SetSkels3dPrev(skelUpdater.GetSkel3d());
		associater.Associate();
		clock_t t_10 = clock();
		skelUpdater.Update(associater.GetSkels2d(), projs);
		std::cout << " 10 use time: " << clock() - t_10 << std::endl;
		std::cout << " associator use time: " << clock() - s_time << std::endl;

		// save
		const int layoutCols = 3;
		std::vector<cv::Rect> rois = SkelPainter::MergeImgs(rawImgs, detectImg, layoutCols,
			{ rawImgs.begin()->cols, rawImgs.begin()->rows });
		detectImg.copyTo(assocImg);
		detectImg.copyTo(reprojImg);

#pragma omp parallel for
		for (int view = 0; view < cameras.size(); view++) {
			//const OpenposeDetection detection = seqDetections[view].Mapping(SKEL19);
			//const OpenposeDetection detection = seqDetections[view];
			//skelPainter.DrawDetect(detection.joints, detection.pafs, detectImg(rois[view]));
			for (const auto& skel2d : associater.GetSkels2d())
				skelPainter.DrawAssoc(skel2d.second.middleCols(view * skelDef.jointSize, skelDef.jointSize), assocImg(rois[view]), skel2d.first);
			for (const auto& skel3d : skelUpdater.GetSkel3d())
				skelPainter.DrawReproj(skel3d.second, projs.middleCols(4 * view, 4), reprojImg(rois[view]), skel3d.first);
		}

		skels.emplace_back(skelUpdater.GetSkel3d());
		cv::imshow("reproj", reprojImg);
		if(cv::waitKey(1) == 'q')
			break;
		std::cout << std::to_string(frameIdx) << std::endl;
	}

	//SerializeSkels(skels, "../output/skel.txt");
	return 0;
}

int main()
{
	run_test();
	//ReadFromSeqDataset();




	return 0;
}


