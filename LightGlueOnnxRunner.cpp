/*********************************
    Copyright: OroChippw
    Author: OroChippw
    Date: 2023.08.31
    Description:
*********************************/
#pragma once
#include <numeric>
extern "C" {
#include "svd.h"
}
#include "LightGlueOnnxRunner.h"
#define Xa 6  //用于计算矩阵
#define Xb 6   //用于计算矩阵
#define ka 6   //用于计算矩阵






float CALCULATE_RANSAC_ERROR(vect2f IMG0_points[50], vect2f IMG1_points[50], TRANSMISSING_Mat* M, int m, int matchNum, float matcherrorthreshold)
{
    if (m == 3)
    {
        float ratio;
        float a11 = M->a11, a12 = M->a12, a21 = M->a21, a22 = M->a22, dx = M->dx, dy = M->dy;
        float x_estimate;
        float y_estimate;
        float x_src, x_dst;
        float y_src, y_dst;
        float distance_error;
        int correct_count = 0;
        for (int i = 0; i < matchNum; i++)
        {
            x_src = IMG0_points[i].x;
            y_src = IMG0_points[i].y;
            x_dst = IMG1_points[i].x;
            y_dst = IMG1_points[i].y;
            x_estimate = a11 * x_src + a12 * y_src + dx;
            y_estimate = a21 * x_src + a22 * y_src + dy;
            distance_error = sqrt(pow(x_estimate - x_dst, 2) + pow(y_estimate - y_dst, 2));
            if (distance_error < matcherrorthreshold)
                correct_count++;
        }
        ratio = (float)correct_count / (float)matchNum;
        return ratio;
    }
    else
    {
        float ratio;
        float a11 = M->a11, a12 = M->a12, a21 = M->a21, a22 = M->a22, dx = M->dx, dy = M->dy, a31 = M->a31, a32 = M->a32;
        float x_estimate;
        float y_estimate;
        float x_src, x_dst;
        float y_src, y_dst;
        float distance_error;
        int correct_count = 0;
        for (int i = 0; i < matchNum; i++)
        {
            x_src = IMG0_points[i].x;
            y_src = IMG0_points[i].y;
            x_dst = IMG1_points[i].x;
            y_dst = IMG1_points[i].y;
            x_estimate = (a11 * x_src + a12 * y_src + dx) / (a31 * x_src + a32 * y_src + 1);
            y_estimate = (a21 * x_src + a22 * y_src + dy) / (a31 * x_src + a32 * y_src + 1);
            distance_error = sqrt(pow(x_estimate - x_dst, 2) + pow(y_estimate - y_dst, 2));
            if (distance_error < matcherrorthreshold)
            {
                correct_count++;
            }

        }
        ratio = (float)correct_count / (float)matchNum;
        return ratio;
    }
}

static void gaussian_elimination_new(float* input, int n)
{
    float* A = input;
    int i = 0;
    int j = 0;
    //m = 8 rows, n = 9 cols
    int m = n - 1;
    while (i < m && j < n)
    {
        // Find pivot in column j, starting in row i:
        int maxi = i;
        for (int k = i + 1; k < m; k++)
        {
            //选取第j列最大的数，并记录行
            if (fabs(A[k * n + j]) > fabs(A[maxi * n + j]))
            {
                maxi = k;
            }
        }
        if (A[maxi * n + j] != 0)
        {
            //swap rows i and maxi, but do not change the value of i
            if (i != maxi)
            {
                for (int k = 0; k < n; k++)
                {
                    float aux = A[i * n + k];
                    A[i * n + k] = A[maxi * n + k];
                    A[maxi * n + k] = aux;
                }
            }
            //Now A[i,j] will contain the old value of A[maxi,j].
            //divide each entry in row i by A[i,j]
            //将主行归一化
            float A_ij = A[i * n + j];
            for (int k = 0; k < n; k++)
            {
                A[i * n + k] /= A_ij;
            }
            //Now A[i,j] will have the value 1.
            //主行*A[u,j]，再用A[u,j]-该数即可消除
            for (int u = i + 1; u < m; u++)
            {
                //subtract A[u,j] * row i from row u
                float A_uj = A[u * n + j];
                for (int k = 0; k < n; k++)
                {
                    A[u * n + k] -= A_uj * A[i * n + k];
                }
                //Now A[u,j] will be 0, since A[u,j] - A[i,j] * A[u,j] = A[u,j] - 1 * A[u,j] = 0.
            }
            i++;
        }
        j++;
    }

    //back substitution
    //最后一位不用管，其他各行用最后一个数-前面各列数*已求的未知数
    for (int i = m - 2; i >= 0; i--)
    {
        for (int j = i + 1; j < n - 1; j++)
        {
            A[i * n + m] -= A[i * n + j] * A[j * n + m];
        }
    }
}


void EVALUATE_M(TRANSMISSING_Mat* M_last, TRANSMISSING_Mat* M, int m, int nm, float matcherrorthreshold, vect2f* image_points0_matched, vect2f* image_points1_matched)
{
    int circle = 0;

    float Max_P_ratio = 0;
    float P_ratio;
    if (m == 3)
    {
        while (circle < 200)  //次数默认200
        {
            vect2f  image_points_sample[2][5];
            //从样本集matched中随机抽选一个RANSAC样本(即一个3个特征点的数组)，放到样本变量sample中  
            //sample = draw_ransac_sample(matched, nm, m);
            //从样本中获取特征点和其对应匹配点的二维坐标，分别放到输出参数pts和mpts中  
            //extract_corresp_pts(sample, m, mtype, &pts, &mpts);
            int index_all[5] = { -1,-1,-1,-1,-1 };
            int z = 0;

            while (z < 5)
            {
                int index = rand() % nm;
                //int index = z + 8;
                if ((index != index_all[0]) && (index != index_all[1]) && (index != index_all[2]) && (index != index_all[3]) && (index != index_all[4]))
                {
                    image_points_sample[0][z] = image_points0_matched[index];
                    image_points_sample[1][z] = image_points1_matched[index];
                    index_all[z] = index;
                    z++;
                }
            }

            ////构建方程组
            //float P[6][7] =
            //{
            //	{image_points_sample[0][0].x, image_points_sample[0][0].y, 0,  0,  1,  0, image_points_sample[1][0].x},
            //	{  0,   0,  image_points_sample[0][0].x, image_points_sample[0][0].y, 0, 1, image_points_sample[1][0].y},
            //	{image_points_sample[0][1].x, image_points_sample[0][1].y, 0,   0,  1,  0, image_points_sample[1][1].x},
            //	{  0,   0,  image_points_sample[0][1].x, image_points_sample[0][1].y, 0, 1, image_points_sample[1][1].y},
            //	{image_points_sample[0][2].x, image_points_sample[0][2].y, 0,   0,  1,  0, image_points_sample[1][2].x},
            //	{  0,   0,  image_points_sample[0][2].x, image_points_sample[0][2].y, 0, 1, image_points_sample[1][2].y}
            //};

            //gaussian_elimination_new(&P[0][0], 7);


            int i, j, p, q;

            double a[Xa * Xb], b[Xa], x[Xb], aa[Xa * Xb], u[Xa * Xa], v[Xb * Xb], eps = 0.0001, SS;
            char ch1[20], ch2[20], flag1, flag2;

            for (i = 0; i < Xa; i++) {
                p = i % 2;
                q = i / 2;
                if (p == 0) {
                    a[i * Xb + 0] = (double)image_points_sample[0][q].x;
                    a[i * Xb + 1] = (double)image_points_sample[0][q].y;
                    a[i * Xb + 2] = 0.;
                    a[i * Xb + 3] = 0.;
                    a[i * Xb + 4] = 1.;
                    a[i * Xb + 5] = 0.;

                    b[i] = (double)image_points_sample[1][q].x;
                }
                else if (p == 1) {
                    a[i * Xb + 0] = 0.;
                    a[i * Xb + 1] = 0.;
                    a[i * Xb + 2] = (double)image_points_sample[0][q].x;
                    a[i * Xb + 3] = (double)image_points_sample[0][q].y;
                    a[i * Xb + 4] = 0.;
                    a[i * Xb + 5] = 1.;

                    b[i] = (double)image_points_sample[1][q].y;
                }
            }

            i = ginv(a, Xa, Xb, aa, eps, u, v, ka);

            for (i = 0; i < Xb; i++)
            {
                SS = 0.0;
                for (j = 0; j < Xa; j++)
                {
                    SS = SS + aa[i * Xa + j] * b[j];
                }
                x[i] = SS;
            }


            M->a11 = (float)x[0];
            M->a12 = (float)x[1];
            M->a21 = (float)x[2];
            M->a22 = (float)x[3];
            M->dx = (float)x[4];
            M->dy = (float)x[5];

            P_ratio = CALCULATE_RANSAC_ERROR(image_points0_matched, image_points1_matched, M, m, nm, matcherrorthreshold);

            if (P_ratio > Max_P_ratio)
            {
                Max_P_ratio = P_ratio;

                M_last->a11 = M->a11;
                M_last->a12 = M->a12;
                M_last->dx = M->dx;
                M_last->a21 = M->a21;
                M_last->a22 = M->a22;
                M_last->dy = M->dy;
            }

            circle++;

        }
    }

    else if (m == 4)
    {
        vect2f  image_points_sample[2][4];
        //从样本集matched中随机抽选一个RANSAC样本(即一个4个特征点的数组)，放到样本变量sample中  
        //sample = draw_ransac_sample(matched, nm, m);
        //从样本中获取特征点和其对应匹配点的二维坐标，分别放到输出参数pts和mpts中  
        //extract_corresp_pts(sample, m, mtype, &pts, &mpts);
        int index_all[4] = { -1,-1,-1,-1 };
        int z = 0;

        while (z < m)
        {
            int index = rand() % nm;
            if ((index != index_all[0]) && (index != index_all[1]) && (index != index_all[2]) && (index != index_all[3]))
            {
                image_points_sample[0][z] = image_points0_matched[index];
                image_points_sample[1][z] = image_points1_matched[index];
                index_all[z] = index;
                z++;
            }
        }


        float P[8][9] =
        {
            {-image_points_sample[0][0].x, -image_points_sample[0][0].y, -1,   0,   0,  0, image_points_sample[0][0].x * image_points_sample[1][0].x, image_points_sample[0][0].y * image_points_sample[1][0].x, -image_points_sample[1][0].x }, // h11
            {  0,   0,  0, -image_points_sample[0][0].x, -image_points_sample[0][0].y, -1, image_points_sample[0][0].x * image_points_sample[1][0].y, image_points_sample[0][0].y * image_points_sample[1][0].y, -image_points_sample[1][0].y }, // h12

            {-image_points_sample[0][1].x, -image_points_sample[0][1].y, -1,   0,   0,  0, image_points_sample[0][1].x * image_points_sample[1][1].x, image_points_sample[0][1].y * image_points_sample[1][1].x, -image_points_sample[1][1].x }, // h13
            {  0,   0,  0, -image_points_sample[0][1].x, -image_points_sample[0][1].y, -1, image_points_sample[0][1].x * image_points_sample[1][1].y, image_points_sample[0][1].y * image_points_sample[1][1].y, -image_points_sample[1][1].y }, // h21

            {-image_points_sample[0][2].x, -image_points_sample[0][2].y, -1,   0,   0,  0, image_points_sample[0][2].x * image_points_sample[1][2].x, image_points_sample[0][2].y * image_points_sample[1][2].x, -image_points_sample[1][2].x }, // h22
            {  0,   0,  0, -image_points_sample[0][2].x, -image_points_sample[0][2].y, -1, image_points_sample[0][2].x * image_points_sample[1][2].y, image_points_sample[0][2].y * image_points_sample[1][2].y, -image_points_sample[1][2].y }, // h23

            {-image_points_sample[0][3].x, -image_points_sample[0][3].y, -1,   0,   0,  0, image_points_sample[0][3].x * image_points_sample[1][3].x, image_points_sample[0][3].y * image_points_sample[1][3].x, -image_points_sample[1][3].x }, // h31
            {  0,   0,  0, -image_points_sample[0][3].x, -image_points_sample[0][3].y, -1, image_points_sample[0][3].x * image_points_sample[1][3].y, image_points_sample[0][3].y * image_points_sample[1][3].y, -image_points_sample[1][3].y }, // h32
        };

        gaussian_elimination_new(&P[0][0], 9);

        M->a11 = P[0][8];
        M->a12 = P[1][8];
        M->dx = P[2][8];
        M->a21 = P[3][8];
        M->a22 = P[4][8];
        M->dy = P[5][8];
        M->a31 = P[6][8];
        M->a32 = P[7][8];

        P_ratio = CALCULATE_RANSAC_ERROR(image_points0_matched, image_points1_matched, M, m, nm, matcherrorthreshold);

        if (P_ratio > Max_P_ratio)
        {
            Max_P_ratio = P_ratio;

            M_last->a11 = P[0][8];
            M_last->a12 = P[1][8];
            M_last->dx = P[2][8];
            M_last->a21 = P[3][8];
            M_last->a22 = P[4][8];
            M_last->dy = P[5][8];

        }

        circle++;
    }

}
int LightGlueOnnxRunner::InitOrtEnv(Configuration cfg)
{
    std::cout << "< - * -------- INITIAL ONNXRUNTIME ENV START -------- * ->" << std::endl;
    try
    {
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LightGlueOnnxRunner");
        session_options = Ort::SessionOptions();
        session_options.SetInterOpNumThreads(std::thread::hardware_concurrency());
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (cfg.device == "cuda") {
            std::cout << "[INFO] OrtSessionOptions Append CUDAExecutionProvider" << std::endl;
            OrtCUDAProviderOptions cuda_options{};

            cuda_options.device_id = 0;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
            cuda_options.gpu_mem_limit = 0;
            cuda_options.arena_extend_strategy = 1; // 设置GPU内存管理中的Arena扩展策略
            cuda_options.do_copy_in_default_stream = 1; // 是否在默认CUDA流中执行数据复制
            cuda_options.has_user_compute_stream = 0;
            cuda_options.default_memory_arena_cfg = nullptr;

            session_options.AppendExecutionProvider_CUDA(cuda_options);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        }

#if _WIN32
        std::cout << "[INFO] Env _WIN32 change modelpath from multi byte to wide char ..." << std::endl;
        const wchar_t* modelPath = multi_Byte_To_Wide_Char(cfg.lightgluePath);
#else
        const char* modelPath = cfg.lightgluePath;
#endif // _WIN32

        session = std::make_unique<Ort::Session>(env, modelPath, session_options);

        const size_t numInputNodes = session->GetInputCount();
        InputNodeNames.reserve(numInputNodes);
        for (size_t i = 0; i < numInputNodes; i++)
        {
            InputNodeNames.emplace_back(_strdup(session->GetInputNameAllocated(i, allocator).get()));
            InputNodeShapes.emplace_back(session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        const size_t numOutputNodes = session->GetOutputCount();
        OutputNodeNames.reserve(numOutputNodes);
        for (size_t i = 0; i < numOutputNodes; i++)
        {
            OutputNodeNames.emplace_back(_strdup(session->GetOutputNameAllocated(i, allocator).get()));
            OutputNodeShapes.emplace_back(session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        std::cout << "[INFO] ONNXRuntime environment created successfully." << std::endl;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[ERROR] ONNXRuntime environment created failed : " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

cv::Mat LightGlueOnnxRunner::PreProcess(Configuration cfg, const cv::Mat& Image, float& scale)
{
    float temp_scale = scale;
    cv::Mat tempImage = Image.clone();
    std::cout << "[INFO] Image info :  width : " << Image.cols << " height :  " << Image.rows << std::endl;

    std::string fn = "max";
    std::string interp = "area";
    tempImage = ResizeImage(tempImage, cfg.image_size, scale, fn, interp);
    cv::Mat resultImage = NormalizeImage(tempImage);
    if (cfg.extractorType == "superpoint")
    {
        std::cout << "[INFO] ExtractorType Superpoint turn RGB to Grayscale" << std::endl;
        resultImage = RGB2Grayscale(resultImage);
    }
    std::cout << "[INFO] Scale from " << temp_scale << " to " << scale << std::endl;

    return resultImage;
}

int LightGlueOnnxRunner::Inference(Configuration cfg, const cv::Mat& src, const cv::Mat& dest)
{
    try
    {
        // Dynamic InputNodeShapes is [1,3,-1,-1]  
        std::cout << "[INFO] srcImage Size : " << src.size() << " Channels : " << src.channels() << std::endl;
        std::cout << "[INFO] destImage Size : " << dest.size() << " Channels : " << src.channels() << std::endl;

        // Build src input node shape and destImage input node shape
        int srcInputTensorSize, destInputTensorSize;
        if (cfg.extractorType == "superpoint")
        {
            //InputNodeShapes[0] = {1 , 1 , src.size().height , src.size().width};
            InputNodeShapes.push_back({ 1 , 1 , src.size().height , src.size().width });
            InputNodeShapes[0] = { 1 , 1 , src.size().height , src.size().width };
            //InputNodeShapes[1] = {1 , 1 , dest.size().height , dest.size().width};
            //InputNodeShapes.push_back({ 1 , 1 , dest.size().height , dest.size().width });
            InputNodeShapes[1] = { 1 , 1 , dest.size().height , dest.size().width };
        }
        else if (cfg.extractorType == "disk")
        {
            InputNodeShapes[0] = { 1 , 3 , src.size().height , src.size().width };
            InputNodeShapes[1] = { 1 , 3 , dest.size().height , dest.size().width };
        }
        srcInputTensorSize = InputNodeShapes[0][0] * InputNodeShapes[0][1] * InputNodeShapes[0][2] * InputNodeShapes[0][3];
        destInputTensorSize = InputNodeShapes[1][0] * InputNodeShapes[1][1] * InputNodeShapes[1][2] * InputNodeShapes[1][3];

        std::vector<float> srcInputTensorValues(srcInputTensorSize);
        std::vector<float> destInputTensorValues(destInputTensorSize);

        if (cfg.extractorType == "superpoint")
        {
            srcInputTensorValues.assign(src.begin<float>(), src.end<float>());
            destInputTensorValues.assign(dest.begin<float>(), dest.end<float>());
        }
        else {
            int src_height = src.rows;
            int src_width = src.cols;
            for (int y = 0; y < src_height; y++) {
                for (int x = 0; x < src_width; x++) {
                    cv::Vec3f pixel = src.at<cv::Vec3f>(y, x); // RGB
                    srcInputTensorValues[y * src_width + x] = pixel[2];
                    srcInputTensorValues[src_height * src_width + y * src_width + x] = pixel[1];
                    srcInputTensorValues[2 * src_height * src_width + y * src_width + x] = pixel[0];
                }
            }
            int dest_height = dest.rows;
            int dest_width = dest.cols;
            for (int y = 0; y < dest_height; y++) {
                for (int x = 0; x < dest_width; x++) {
                    cv::Vec3f pixel = dest.at<cv::Vec3f>(y, x);
                    destInputTensorValues[y * dest_width + x] = pixel[2];
                    destInputTensorValues[dest_height * dest_width + y * dest_width + x] = pixel[1];
                    destInputTensorValues[2 * dest_height * dest_width + y * dest_width + x] = pixel[0];
                }
            }
        }

        auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler, srcInputTensorValues.data(), srcInputTensorValues.size(), \
            InputNodeShapes[0].data(), InputNodeShapes[0].size()
        ));

        auto time_start = std::chrono::high_resolution_clock::now();

        //const char* const* input_names, const Value* input_values, size_t input_count,const char* const* output_names, Value* output_values, size_t output_count
        auto output_tensor = session->Run(Ort::RunOptions{ nullptr }, InputNodeNames.data(), input_tensors.data(), \
            input_tensors.size(), OutputNodeNames.data(), OutputNodeNames.size());

        auto time_end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
        timer += diff;

        for (auto& tensor : output_tensor)
        {
            if (!tensor.IsTensor() || !tensor.HasValue())
            {
                std::cerr << "[ERROR] Inference output tensor is not a tensor or don't have value" << std::endl;
            }
        }
        output_tensors = std::move(output_tensor);


        std::vector<Ort::Value> input_tensors1;
        input_tensors1.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler, destInputTensorValues.data(), destInputTensorValues.size(), \
            InputNodeShapes[1].data(), InputNodeShapes[1].size()
        ));
        

        auto output_tensor1 = session->Run(Ort::RunOptions{ nullptr }, InputNodeNames.data(), input_tensors1.data(), \
            input_tensors1.size(), OutputNodeNames.data(), OutputNodeNames.size());

        output_tensors1 = std::move(output_tensor1);
       



        std::cout << "[INFO] LightGlueOnnxRunner inference finish ..." << std::endl;
        std::cout << "[INFO] Inference cost time : " << diff << "ms" << std::endl;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[ERROR] LightGlueOnnxRunner inference failed : " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
//std::vector<cv::KeyPoint> getKeyPoints(const std::vector<Ort::OrtSessionHandler::DataOutputType>& inferenceOutput, int borderRemove,
//    float confidenceThresh) const
//{
//    std::vector<int> detectorShape(inferenceOutput[0].second.begin() + 1, inferenceOutput[0].second.end());
//
//    cv::Mat detectorMat(detectorShape.size(), detectorShape.data(), CV_32F,
//        inferenceOutput[0].first);  // 65 x H/8 x W/8
//    cv::Mat buffer;
//
//    transposeNDWrapper(detectorMat, { 1, 2, 0 }, buffer);
//    buffer.copyTo(detectorMat);  // H/8 x W/8 x 65
//
//    for (int i = 0; i < detectorShape[1]; ++i) {
//        for (int j = 0; j < detectorShape[2]; ++j) {
//            Ort::softmax(detectorMat.ptr<float>(i, j), detectorShape[0]);
//        }
//    }
//    detectorMat = detectorMat({ cv::Range::all(), cv::Range::all(), cv::Range(0, detectorShape[0] - 1) })
//        .clone();                                                        // H/8 x W/8 x 64
//    detectorMat = detectorMat.reshape(1, { detectorShape[1], detectorShape[2], 8, 8 });  // H/8 x W/8 x 8 x 8
//    transposeNDWrapper(detectorMat, { 0, 2, 1, 3 }, buffer);
//    buffer.copyTo(detectorMat);  // H/8 x 8 x W/8 x 8
//
//    detectorMat = detectorMat.reshape(1, { detectorShape[1] * 8, detectorShape[2] * 8 });  // H x W
//
//    std::vector<cv::KeyPoint> keyPoints;
//    for (int i = borderRemove; i < detectorMat.rows - borderRemove; ++i) {
//        auto rowPtr = detectorMat.ptr<float>(i);
//        for (int j = borderRemove; j < detectorMat.cols - borderRemove; ++j) {
//            if (rowPtr[j] > confidenceThresh) {
//                cv::KeyPoint keyPoint;
//                keyPoint.pt.x = j;
//                keyPoint.pt.y = i;
//                keyPoint.response = rowPtr[j];
//                keyPoints.emplace_back(keyPoint);
//            }
//        }
//    }
//
//    return keyPoints;
//}

inline void transposeNDWrapper(cv::InputArray src_, const std::vector<int>& order, cv::OutputArray dst_)
{
#if (CV_MAJOR_VERSION > 4 || (CV_MAJOR_VERSION == 4 && CV_MINOR_VERSION >= 6))
    cv::transposeND(src_, order, dst_);
#else
    cv::Mat inp = src_.getMat();
    CV_Assert(inp.isContinuous());
    CV_CheckEQ(inp.channels(), 1, "Input array should be single-channel");
    CV_CheckEQ(order.size(), static_cast<size_t>(inp.dims), "Number of dimensions shouldn't change");

    auto order_ = order;
    std::sort(order_.begin(), order_.end());
    for (size_t i = 0; i < order_.size(); ++i) {
        CV_CheckEQ(static_cast<size_t>(order_[i]), i, "New order should be a valid permutation of the old one");
    }

    std::vector<int> newShape(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        newShape[i] = inp.size[order[i]];
    }

    dst_.create(static_cast<int>(newShape.size()), newShape.data(), inp.type());
    cv::Mat out = dst_.getMat();
    CV_Assert(out.isContinuous());
    CV_Assert(inp.data != out.data);

    int continuous_idx = 0;
    for (int i = static_cast<int>(order.size()) - 1; i >= 0; --i) {
        if (order[i] != i) {
            continuous_idx = i + 1;
            break;
        }
    }

    size_t continuous_size = continuous_idx == 0 ? out.total() : out.step1(continuous_idx - 1);
    size_t outer_size = out.total() / continuous_size;

    std::vector<size_t> steps(order.size());
    for (int i = 0; i < static_cast<int>(steps.size()); ++i) {
        steps[i] = inp.step1(order[i]);
    }

    auto* src = inp.ptr<const unsigned char>();
    auto* dst = out.ptr<unsigned char>();

    size_t src_offset = 0;
    size_t es = out.elemSize();
    for (size_t i = 0; i < outer_size; ++i) {
        std::memcpy(dst, src + es * src_offset, es * continuous_size);
        dst += es * continuous_size;
        for (int j = continuous_idx - 1; j >= 0; --j) {
            src_offset += steps[j];
            if ((src_offset / steps[j]) % out.size[j] != 0) {
                break;
            }
            src_offset -= steps[j] * out.size[j];
        }
    }
#endif
}

inline void softmax(float* input, const size_t inputLen)
{
    const float maxVal = *std::max_element(input, input + inputLen);

    const float sum = std::accumulate(input, input + inputLen, 0.0,
        [&](float a, const float b) { return std::move(a) + expf(b - maxVal); });

    const float offset = maxVal + logf(sum);
    for (auto it = input; it != (input + inputLen); ++it) {
        *it = expf(*it - offset);
    }
}

std::vector<int> nmsFast(const std::vector<cv::KeyPoint>& keyPoints, int height, int width,int distThresh) 
{
    static const int TO_PROCESS = 1;
    static const int EMPTY_OR_SUPPRESSED = 0;

    std::vector<int> sortedIndices(keyPoints.size());
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);

    // sort in descending order base on confidence
    std::stable_sort(sortedIndices.begin(), sortedIndices.end(),
        [&keyPoints](int lidx, int ridx) { return keyPoints[lidx].response > keyPoints[ridx].response; });

    cv::Mat grid = cv::Mat(height, width, CV_8U, TO_PROCESS);
    std::vector<int> keepIndices;

    for (int idx : sortedIndices) {
        int x = keyPoints[idx].pt.x;
        int y = keyPoints[idx].pt.y;

        if (grid.at<uchar>(y, x) == TO_PROCESS) {
            for (int i = y - distThresh; i < y + distThresh; ++i) {
                if (i < 0 || i >= height) {
                    continue;
                }

                for (int j = x - distThresh; j < x + distThresh; ++j) {
                    if (j < 0 || j >= width) {
                        continue;
                    }
                    grid.at<uchar>(i, j) = EMPTY_OR_SUPPRESSED;
                }
            }
            keepIndices.emplace_back(idx);
        }
    }

    return keepIndices;
}

inline cv::Mat bilinearGridSample(const cv::Mat& input, const cv::Mat& grid, bool alignCorners)
{
    // input: B x C x Hi x Wi
    // grid: B x Hg x Wg x 2

    if (input.size[0] != grid.size[0]) {
        throw std::runtime_error("input and grid need to have the same batch size");
    }
    int batch = input.size[0];
    int channel = input.size[1];
    int height = input.size[2];
    int width = input.size[3];

    int numKeyPoints = grid.size[2];
    cv::Mat yMat = grid({ cv::Range::all(), cv::Range::all(), cv::Range::all(), cv::Range(0, 1) })
        .clone()
        .reshape(1, { batch, grid.size[1] * grid.size[2] });
    cv::Mat xMat = grid({ cv::Range::all(), cv::Range::all(), cv::Range::all(), cv::Range(1, 2) })
        .clone()
        .reshape(1, { batch, grid.size[1] * grid.size[2] });

    if (alignCorners) {
        xMat = ((xMat + 1) / 2) * (width - 1);
        yMat = ((yMat + 1) / 2) * (height - 1);
    }
    else {
        xMat = ((xMat + 1) * width - 1) / 2;
        yMat = ((yMat + 1) * height - 1) / 2;
    }

    // floor
    cv::Mat x0Mat = xMat - 0.5;
    cv::Mat y0Mat = yMat - 0.5;
    x0Mat.convertTo(x0Mat, CV_32S);
    y0Mat.convertTo(y0Mat, CV_32S);

    x0Mat.convertTo(x0Mat, CV_32F);
    y0Mat.convertTo(y0Mat, CV_32F);

    cv::Mat x1Mat = x0Mat + 1;
    cv::Mat y1Mat = x0Mat + 1;

    std::vector<cv::Mat> weights = { (x1Mat - xMat).mul(y1Mat - yMat), (x1Mat - xMat).mul(yMat - y0Mat),
                                    (xMat - x0Mat).mul(y1Mat - yMat), (xMat - x0Mat).mul(yMat - y0Mat) };

    cv::Mat result = cv::Mat::zeros(3, std::vector<int>{batch, channel, grid.size[1] * grid.size[2]}.data(), CV_32F);

    auto isCoordSafe = [](int size, int maxSize) -> bool { return size > 0 && size < maxSize; };

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < grid.size[1] * grid.size[2]; ++i) {
            int x0 = x0Mat.at<float>(b, i);
            int y0 = y0Mat.at<float>(b, i);
            int x1 = x1Mat.at<float>(b, i);
            int y1 = y1Mat.at<float>(b, i);

            std::vector<std::pair<int, int>> pairs = { {x0, y0}, {x0, y1}, {x1, y0}, {x1, y1} };
            std::vector<cv::Mat> Is(4, cv::Mat::zeros(channel, 1, CV_32F));

            for (int k = 0; k < 4; ++k) {
                if (isCoordSafe(pairs[k].first, width) && isCoordSafe(pairs[k].second, height)) {
                    Is[k] =
                        input({ cv::Range(b, b + 1), cv::Range::all(), cv::Range(pairs[k].second, pairs[k].second + 1),
                               cv::Range(pairs[k].first, pairs[k].first + 1) })
                        .clone()
                        .reshape(1, channel);
                }
            }

            cv::Mat curDescriptor = Is[0] * weights[0].at<float>(i) + Is[1] * weights[1].at<float>(i) +
                Is[2] * weights[2].at<float>(i) + Is[3] * weights[3].at<float>(i);

            for (int c = 0; c < channel; ++c) {
                result.at<float>(b, c, i) = curDescriptor.at<float>(c);
            }
        }
    }

    return result.reshape(1, { batch, channel, grid.size[1], grid.size[2] });
}


cv::Mat getDescriptors(const cv::Mat& coarseDescriptors, const std::vector<cv::KeyPoint>& keyPoints,
    int height, int width, bool alignCorners) 
{
    cv::Mat keyPointMat(keyPoints.size(), 2, CV_32F);

    for (int i = 0; i < keyPoints.size(); ++i) {
        auto rowPtr = keyPointMat.ptr<float>(i);
        rowPtr[0] = 2 * keyPoints[i].pt.y / (height - 1) - 1;
        rowPtr[1] = 2 * keyPoints[i].pt.x / (width - 1) - 1;
    }
    keyPointMat = keyPointMat.reshape(1, { 1, 1, static_cast<int>(keyPoints.size()), 2 });
    cv::Mat descriptors = bilinearGridSample(coarseDescriptors, keyPointMat, alignCorners);
    descriptors = descriptors.reshape(1, { coarseDescriptors.size[1], static_cast<int>(keyPoints.size()) });

    cv::Mat buffer;
    transposeNDWrapper(descriptors, { 1, 0 }, buffer);

    return buffer;
}

int LightGlueOnnxRunner::PostProcess(Configuration cfg, int origW, int origH, int reshapeW, int reshapeH, cv::Mat image0, cv::Mat image1 )
{
    try {

        std::vector<int64_t> image0_output0_Shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        float* image0_output0_matrix = (float*)output_tensors[0].GetTensorMutableData<void>();


        int detectorShape[3] = { 65,reshapeH / 8,reshapeW / 8 };
        cv::Mat detectorMat(3, detectorShape, CV_32F, image0_output0_matrix);

        cv::Mat buffer;
        transposeNDWrapper(detectorMat, { 1, 2, 0 }, buffer);
        buffer.copyTo(detectorMat);  // H/8 x W/8 x 65

        for (int i = 0; i < detectorShape[1]; ++i) {
            for (int j = 0; j < detectorShape[2]; ++j) {
                softmax(detectorMat.ptr<float>(i, j), detectorShape[0]);
            }
        }

        detectorMat = detectorMat({ cv::Range::all(), cv::Range::all(), cv::Range(0, detectorShape[0] - 1) })
            .clone();                                                        // H/8 x W/8 x 64
        detectorMat = detectorMat.reshape(1, { detectorShape[1], detectorShape[2], 8, 8 });  // H/8 x W/8 x 8 x 8
        transposeNDWrapper(detectorMat, { 0, 2, 1, 3 }, buffer);
        buffer.copyTo(detectorMat);  // H/8 x 8 x W/8 x 8

        detectorMat = detectorMat.reshape(1, { detectorShape[1] * 8, detectorShape[2] * 8 });  // H x W

        std::vector<cv::KeyPoint> keyPoints0;
        int borderRemove = 0;
        float confidenceThresh = 0.25;
        for (int i = borderRemove; i < detectorMat.rows - borderRemove; ++i) {
            auto rowPtr = detectorMat.ptr<float>(i);
            for (int j = borderRemove; j < detectorMat.cols - borderRemove; ++j) {
                if (rowPtr[j] > confidenceThresh) {
                    cv::KeyPoint keyPoint;
                    keyPoint.pt.x = j;
                    keyPoint.pt.y = i;
                    keyPoint.response = rowPtr[j];
                    keyPoints0.emplace_back(keyPoint);
                }
            }
        }
        std::vector<int> keepIndices = nmsFast(keyPoints0, reshapeH, reshapeW, 0.1);

        std::vector<int64_t> image0_output1_Shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
        float* image0_output1_matrix = (float*)output_tensors[1].GetTensorMutableData<void>();
        int descriptorShape[4] = { 1,256,reshapeH / 8,reshapeW / 8 };
        cv::Mat coarseDescriptorMat(4, descriptorShape, CV_32F, image0_output1_matrix);  // 1 x 256 x H/8 x W/8

        std::vector<cv::KeyPoint> keepKeyPoints;
        keepKeyPoints.reserve(keepIndices.size());
        std::transform(keepIndices.begin(), keepIndices.end(), std::back_inserter(keepKeyPoints),
            [&keyPoints0](int idx) { return keyPoints0[idx]; });
        keyPoints0 = std::move(keepKeyPoints);

        cv::Mat descriptors = getDescriptors(coarseDescriptorMat, keyPoints0, reshapeH, reshapeW, true);  //alignCorners is true

        for (auto& keyPoint : keyPoints0) {
            keyPoint.pt.x *= static_cast<float>(origW) / reshapeW;
            keyPoint.pt.y *= static_cast<float>(origH) / reshapeH;
        }

        //**************************************第二幅图提取特征点和特征向量*****************************//

        std::vector<int64_t> image1_output0_Shape = output_tensors1[0].GetTensorTypeAndShapeInfo().GetShape();
        float* image1_output0_matrix = (float*)output_tensors1[0].GetTensorMutableData<void>();

        int detectorShape1[3] = { 65,reshapeH / 8,reshapeW / 8 };
        cv::Mat detectorMat1(3, detectorShape1, CV_32F, image1_output0_matrix);

        cv::Mat buffer1;
        transposeNDWrapper(detectorMat1, { 1, 2, 0 }, buffer1);
        buffer1.copyTo(detectorMat1);  // H/8 x W/8 x 65

        for (int i = 0; i < detectorShape1[1]; ++i) {
            for (int j = 0; j < detectorShape1[2]; ++j) {
                softmax(detectorMat1.ptr<float>(i, j), detectorShape1[0]);
            }
        }

        detectorMat1 = detectorMat1({ cv::Range::all(), cv::Range::all(), cv::Range(0, detectorShape1[0] - 1) })
            .clone();                                                        // H/8 x W/8 x 64
        detectorMat1 = detectorMat1.reshape(1, { detectorShape1[1], detectorShape1[2], 8, 8 });  // H/8 x W/8 x 8 x 8
        transposeNDWrapper(detectorMat1, { 0, 2, 1, 3 }, buffer1);
        buffer1.copyTo(detectorMat1);  // H/8 x 8 x W/8 x 8

        detectorMat1 = detectorMat1.reshape(1, { detectorShape1[1] * 8, detectorShape1[2] * 8 });  // H x W
        std::vector<cv::KeyPoint> keyPoints1;

        for (int i = borderRemove; i < detectorMat1.rows - borderRemove; ++i) {
            auto rowPtr = detectorMat1.ptr<float>(i);
            for (int j = borderRemove; j < detectorMat1.cols - borderRemove; ++j) {
                if (rowPtr[j] > confidenceThresh) {
                    cv::KeyPoint keyPoint;
                    keyPoint.pt.x = j;
                    keyPoint.pt.y = i;
                    keyPoint.response = rowPtr[j];
                    keyPoints1.emplace_back(keyPoint);
                }
            }
        }
        std::vector<int> keepIndices1 = nmsFast(keyPoints1, reshapeH, reshapeW, 0.1);

        std::vector<int64_t> image1_output1_Shape = output_tensors1[1].GetTensorTypeAndShapeInfo().GetShape();
        float* image1_output1_matrix = (float*)output_tensors1[1].GetTensorMutableData<void>();
        int descriptorShape1[4] = { 1,256,reshapeH / 8,reshapeW / 8 };
        cv::Mat coarseDescriptorMat1(4, descriptorShape1, CV_32F, image1_output1_matrix);  // 1 x 256 x H/8 x W/8

        std::vector<cv::KeyPoint> keepKeyPoints1;
        keepKeyPoints1.reserve(keepIndices1.size());
        std::transform(keepIndices1.begin(), keepIndices1.end(), std::back_inserter(keepKeyPoints1),
            [&keyPoints1](int idx) { return keyPoints1[idx]; });
        keyPoints1 = std::move(keepKeyPoints1);

        cv::Mat descriptors1 = getDescriptors(coarseDescriptorMat1, keyPoints1, reshapeH, reshapeW, true);  //alignCorners is true

        for (auto& keyPoint : keyPoints1) {
            keyPoint.pt.x *= static_cast<float>(origW) / reshapeW;
            keyPoint.pt.y *= static_cast<float>(origH) / reshapeH;
        }

        //***********************************匹配*******************************************************//
        cv::BFMatcher matcher(cv::NORM_L2, true);
        std::vector<cv::DMatch> knnMatches;
        matcher.match(descriptors, descriptors1, knnMatches);

        cv::Mat matchesImage;
        cv::drawMatches(image0, keyPoints0, image1, keyPoints1, knnMatches, matchesImage,
            cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
            cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imwrite("super_point_good_matches.jpg", matchesImage);
     /*   cv::imshow("super_point_good_matches", matchesImage);
        cv::waitKey();*/

        //***********************************求仿射变化矩阵**********************************************//
 
        std::vector<cv::Point2f> kpts_f1;
        std::vector<cv::Point2f> kpts_f2;
        for (int i = 0; i < knnMatches.size(); i += 1)
        {
            if (knnMatches[i].distance < 0.15)
            {
                kpts_f1.emplace_back(cv::Point2f(keyPoints0[knnMatches[i].queryIdx].pt.x, keyPoints0[knnMatches[i].queryIdx].pt.y));
                kpts_f2.emplace_back(cv::Point2f(keyPoints1[knnMatches[i].trainIdx].pt.x, keyPoints1[knnMatches[i].trainIdx].pt.y));
            }
        }

        keypoints_result = { kpts_f1 ,kpts_f2 };

   
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[ERROR] PostProcess failed : " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> LightGlueOnnxRunner::InferenceImage(Configuration cfg,
    const cv::Mat& srcImage, const cv::Mat& destImage)
{
    std::cout << "< - * -------- INFERENCEIMAGE START -------- * ->" << std::endl;

    if (srcImage.empty() || destImage.empty())
    {
        throw  "[ERROR] ImageEmptyError ";
    }
    cv::Mat srcImage_copy = cv::Mat(srcImage);
    cv::Mat destImage_copy = cv::Mat(destImage);

    std::cout << "[INFO] => PreProcess srcImage" << std::endl;
    cv::Mat src = PreProcess(cfg, srcImage_copy, scales[0]);
    std::cout << "[INFO] => PreProcess destImage" << std::endl;
    cv::Mat dest = PreProcess(cfg, destImage_copy, scales[1]);

    Inference(cfg, src, dest);

    int origW = srcImage_copy.cols, origH = srcImage_copy.rows;
    int reshapeW = src.cols, reshapeH = src.rows;


    PostProcess(cfg, origW, origH, reshapeW, reshapeH, srcImage_copy, destImage_copy);

    output_tensors.clear();
    output_tensors1.clear();

    return GetKeypointsResult();
}

float LightGlueOnnxRunner::GetMatchThresh()
{
    return this->matchThresh;
}

void LightGlueOnnxRunner::SetMatchThresh(float thresh)
{
    this->matchThresh = thresh;
}

double LightGlueOnnxRunner::GetTimer(std::string name = "matcher")
{
    return this->timer;
}

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> LightGlueOnnxRunner::GetKeypointsResult()
{
    return this->keypoints_result;
}

LightGlueOnnxRunner::LightGlueOnnxRunner(unsigned int threads) : \
num_threads(threads)
{
}

LightGlueOnnxRunner::~LightGlueOnnxRunner()
{
}
