#include "torch/torch.h"
#include "torch/script.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <string.h>
#include "Yolov5.h"
#include "DataUtils.h"
#include <yaml-cpp/yaml.h>
#include <experimental/filesystem>
#include <stdio.h>

namespace YAML {
	template<>
	struct convert<Yolov5::ROI> {
		static bool decode(const Node& node, Yolov5::ROI& cType) {
			cType.x1 = node["x1"].as<int>();
			cType.y1 = node["y1"].as<int>();
			cType.x2 = node["x2"].as<int>();
			cType.y2 = node["y2"].as<int>();

			return true;
		}
	};
}

int main() {
	//DataUtils::Untar("D:/Downloads/S1_L1.tar.bz2");
	// Loading  Module
	YAML::Node config = YAML::LoadFile("config.yml");

	YAML::Node inputFilesNode = config["input_files"];
	YAML::Node ROIs = config["ROIs"];

	std::vector<Yolov5::ROI> ROIsList;
	std::vector<std::string> ROIsListKeys;
	std::vector<std::string> inputFiles;

	
	for (YAML::const_iterator it = inputFilesNode.begin(); it != inputFilesNode.end(); ++it) {
		inputFiles.push_back(it->as<std::string>());
	}

	
	for (YAML::const_iterator it = ROIs.begin(); it != ROIs.end(); ++it) {
		ROIsListKeys.push_back(it->first.as<std::string>());       // <- key
		ROIsList.push_back(it->second.as<Yolov5::ROI>()); // <- value
	}
	std::string INPUT_PATH = config["input_path"].as<std::string>();
	std::string OUTPUT_PATH = config["output_path"].as<std::string>();

	std::string inputPath;
	std::string outputPath;
	Yolov5 yolov5 =  Yolov5();
	for (std::string file : inputFiles) {
		inputPath = INPUT_PATH + file + "/frame_%04d.jpg";
		outputPath = OUTPUT_PATH + file;
		std::experimental::filesystem::create_directories(outputPath.c_str());
		//_mkdir(outputPath.c_str());
		yolov5.detect(inputPath,outputPath,ROIsListKeys,ROIsList);
	}

	return 0;
}
