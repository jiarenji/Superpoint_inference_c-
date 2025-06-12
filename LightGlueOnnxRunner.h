#pragma once
/*********************************
    Copyright: OroChippw
    Author: OroChippw
    Date: 2023.08.31
    Description:
*********************************/
#pragma once
#pragma warning(disable:4819) 

#ifndef LIGHTGLUE_ONNX_RUNNER_H
#define LIGHTGLUE_ONNX_RUNNER_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
// #include <cuda_provider_factory.h>  // 若在GPU环境下运行可以使用cuda进行加速

#include "utils.h"
#include "transform.h"
#include "BaseOnnxRunner.h"
#include "Configuration.h"
typedef struct {
	float a11;
	float a12;
	float a21;
	float a22;
	float a31;
	float a32;
	float dx;
	float dy;
}TRANSMISSING_Mat;


typedef struct
{
    float x;
    float y;
}vect2f;

void EVALUATE_M(TRANSMISSING_Mat* M_last, TRANSMISSING_Mat* M, int m, int nm, float matcherrorthreshold, vect2f* image_points0_matched, vect2f* image_points1_matched);


class LightGlueOnnxRunner : public BaseFeatureMatchOnnxRunner
{
private:
    const unsigned int num_threads;

    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<char*> InputNodeNames;
    std::vector<std::vector<int64_t>> InputNodeShapes;

    std::vector<char*> OutputNodeNames;
    std::vector<std::vector<int64_t>> OutputNodeShapes;

    float matchThresh = 0.0f;
    long long timer = 0.0f;
    std::vector<float> scales = { 1.0f , 1.0f };

    std::vector<Ort::Value> output_tensors;
    std::vector<Ort::Value> output_tensors1;
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> keypoints_result;


private:
    cv::Mat PreProcess(Configuration cfg, const cv::Mat& srcImage, float& scale);
    int Inference(Configuration cfg, const cv::Mat& src, const cv::Mat& dest);
    int PostProcess(Configuration cfg, int origW, int origH, int reshapeW, int reshapeH, cv::Mat image0, cv::Mat image1);
    //TRANSMISSING_Mat* PostProcess(Configuration cfg, int origW, int origH, int reshapeW, int reshapeH, cv::Mat image0, cv::Mat image1)


public:
    explicit LightGlueOnnxRunner(unsigned int num_threads = 1);
    ~LightGlueOnnxRunner();

    float GetMatchThresh();
    void SetMatchThresh(float thresh);
    double GetTimer(std::string name);

    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> GetKeypointsResult();

    int InitOrtEnv(Configuration cfg);

    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> InferenceImage(Configuration cfg, \
        const cv::Mat& srcImage, const cv::Mat& destImage);

};

#endif // LIGHTGLUE_ONNX_RUNNER_H