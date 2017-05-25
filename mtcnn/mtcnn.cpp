// caffe
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>

// c++
#include <fstream> 
#include <string>
#include <string>
#include <vector>
#include <fstream>
#include <iosfwd>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <utility>
#include <fstream>
// opencv
#include <opencv2/opencv.hpp>
// boost
#include "boost/make_shared.hpp"

//#define CPU_ONLY
#define INTER_FAST
using namespace caffe;
using namespace cv;

typedef struct FaceRect {
  float x1;
  float y1;
  float x2;
  float y2;
  float score; /**< Larger score should mean higher confidence. */
} FaceRect;

typedef struct FacePts {
  float x[5],y[5];
} FacePts;

typedef struct FaceInfo {
  FaceRect bbox;
  cv::Vec4f regression;
  FacePts facePts;
  double roll;
  double pitch;
  double yaw;
} FaceInfo;

class MTCNN {
 public:
  MTCNN(const string& proto_model_dir);
  void Detect(const cv::Mat& img, std::vector<FaceInfo> &faceInfo, int minSize, double* threshold, double factor);
  vector<cv::Mat> getFaceRects(const cv::Mat& img,vector<Rect>* faceRects);       //给定一张图片，将其中人脸ROI返回,以及对应的五个特征点

 private:
  bool CvMatToDatumSignalChannel(const cv::Mat& cv_mat, Datum* datum);
  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);
  void WrapInputLayer(std::vector<cv::Mat>* input_channels,Blob<float>* input_layer,
          const int height,const int width);
  void SetMean();
  void GenerateBoundingBox( Blob<float>* confidence,Blob<float>* reg,
          float scale,float thresh,int image_width,int image_height);
  void ClassifyFace(const std::vector<FaceInfo>& regressed_rects,cv::Mat &sample_single,
        boost::shared_ptr<Net<float> >& net,double thresh,char netName);
  void ClassifyFace_MulImage(const std::vector<FaceInfo> &regressed_rects, cv::Mat &sample_single,
        boost::shared_ptr<Net<float> >& net, double thresh, char netName);
  std::vector<FaceInfo> NonMaximumSuppression(std::vector<FaceInfo>& bboxes,float thresh,char methodType);
  void Bbox2Square(std::vector<FaceInfo>& bboxes);
  void Padding(int img_w, int img_h);
  std::vector<FaceInfo> BoxRegress(std::vector<FaceInfo> &faceInfo_, int stage);
  void RegressPoint(const std::vector<FaceInfo>& faceInfo);

 private:
  boost::shared_ptr<Net<float> > PNet_;
  boost::shared_ptr<Net<float> > RNet_;
  boost::shared_ptr<Net<float> > ONet_;

  // x1,y1,x2,t2 and score
  std::vector<FaceInfo> condidate_rects_;
  std::vector<FaceInfo> total_boxes_;
  std::vector<FaceInfo> regressed_rects_;
  std::vector<FaceInfo> regressed_pading_;

  std::vector<cv::Mat> crop_img_;
  int curr_feature_map_w_;
  int curr_feature_map_h_;
  int num_channels_;
};

// compare score
bool CompareBBox(const FaceInfo & a, const FaceInfo & b) {
  return a.bbox.score > b.bbox.score;
}

// methodType : u is IoU(Intersection Over Union)
// methodType : m is IoM(Intersection Over Maximum)
std::vector<FaceInfo> MTCNN::NonMaximumSuppression(std::vector<FaceInfo>& bboxes,
                float thresh,char methodType){
  std::vector<FaceInfo> bboxes_nms;
  std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

  int32_t select_idx = 0;
  int32_t num_bbox = static_cast<int32_t>(bboxes.size());
  std::vector<int32_t> mask_merged(num_bbox, 0);
  bool all_merged = false;

  while (!all_merged) {
    while (select_idx < num_bbox && mask_merged[select_idx] == 1)
      select_idx++;
    if (select_idx == num_bbox) {
      all_merged = true;
      continue;
    }

    bboxes_nms.push_back(bboxes[select_idx]);
    mask_merged[select_idx] = 1;

    FaceRect select_bbox = bboxes[select_idx].bbox;
    float area1 = static_cast<float>((select_bbox.x2-select_bbox.x1+1) * (select_bbox.y2-select_bbox.y1+1));
    float x1 = static_cast<float>(select_bbox.x1);
    float y1 = static_cast<float>(select_bbox.y1);
    float x2 = static_cast<float>(select_bbox.x2);
    float y2 = static_cast<float>(select_bbox.y2);

    select_idx++;
    for (int32_t i = select_idx; i < num_bbox; i++) {
      if (mask_merged[i] == 1)
        continue;

      FaceRect& bbox_i = bboxes[i].bbox;
      float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
      float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
      float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;
      float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
      if (w <= 0 || h <= 0)
        continue;

      float area2 = static_cast<float>((bbox_i.x2-bbox_i.x1+1) * (bbox_i.y2-bbox_i.y1+1));
      float area_intersect = w * h;

      switch (methodType) {
        case 'u':
          if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > thresh)
            mask_merged[i] = 1;
          break;
        case 'm':
          if (static_cast<float>(area_intersect) / std::min(area1 , area2) > thresh)
            mask_merged[i] = 1;
          break;
        default:
          break;
        }
    }
  }
  return bboxes_nms;
}

void MTCNN::Bbox2Square(std::vector<FaceInfo>& bboxes){
  for(int i=0;i<bboxes.size();i++){
    float h = bboxes[i].bbox.x2 - bboxes[i].bbox.x1;
    float w = bboxes[i].bbox.y2 - bboxes[i].bbox.y1;
    float side = h>w ? h:w;
    bboxes[i].bbox.x1 += (h-side)*0.5;
    bboxes[i].bbox.y1 += (w-side)*0.5;

    bboxes[i].bbox.x2 = (int)(bboxes[i].bbox.x1 + side);
    bboxes[i].bbox.y2 = (int)(bboxes[i].bbox.y1 + side);
    bboxes[i].bbox.x1 = (int)(bboxes[i].bbox.x1);
    bboxes[i].bbox.y1 = (int)(bboxes[i].bbox.y1);

  }
}

std::vector<FaceInfo> MTCNN::BoxRegress(std::vector<FaceInfo>& faceInfo,int stage){
  std::vector<FaceInfo> bboxes;
  for(int bboxId =0;bboxId<faceInfo.size();bboxId++){
      FaceRect faceRect;
      FaceInfo tempFaceInfo;
      float regw = faceInfo[bboxId].bbox.y2 - faceInfo[bboxId].bbox.y1;
      regw += (stage == 1)? 0:1;
      float regh = faceInfo[bboxId].bbox.x2 - faceInfo[bboxId].bbox.x1;
      regh += (stage == 1)? 0:1;
      faceRect.y1 = faceInfo[bboxId].bbox.y1 + regw * faceInfo[bboxId].regression[0];
      faceRect.x1 = faceInfo[bboxId].bbox.x1 + regh * faceInfo[bboxId].regression[1];
      faceRect.y2 = faceInfo[bboxId].bbox.y2 + regw * faceInfo[bboxId].regression[2];
      faceRect.x2 = faceInfo[bboxId].bbox.x2 + regh * faceInfo[bboxId].regression[3];
      faceRect.score = faceInfo[bboxId].bbox.score;

      tempFaceInfo.bbox = faceRect;
      tempFaceInfo.regression = faceInfo[bboxId].regression;
      if(stage == 3)
        tempFaceInfo.facePts = faceInfo[bboxId].facePts;
      bboxes.push_back(tempFaceInfo);
  }
  return bboxes;
}

// compute the padding coordinates (pad the bounding boxes to square)
void MTCNN::Padding(int img_w,int img_h){
  for(int i=0;i<regressed_rects_.size();i++){
    FaceInfo tempFaceInfo;
    tempFaceInfo = regressed_rects_[i];
    tempFaceInfo.bbox.y2 = (regressed_rects_[i].bbox.y2 >= img_w) ? img_w : regressed_rects_[i].bbox.y2;
    tempFaceInfo.bbox.x2 = (regressed_rects_[i].bbox.x2 >= img_h) ? img_h : regressed_rects_[i].bbox.x2;
    tempFaceInfo.bbox.y1 = (regressed_rects_[i].bbox.y1 <1) ? 1 : regressed_rects_[i].bbox.y1;
    tempFaceInfo.bbox.x1 = (regressed_rects_[i].bbox.x1 <1) ? 1 : regressed_rects_[i].bbox.x1;
    regressed_pading_.push_back(tempFaceInfo);
  }
}

void MTCNN::GenerateBoundingBox(Blob<float>* confidence,Blob<float>* reg,
      float scale,float thresh,int image_width,int image_height){
  int stride = 2;
  int cellSize = 12;

  int curr_feature_map_w_ = std::ceil((image_width - cellSize)*1.0/stride)+1;
  int curr_feature_map_h_ = std::ceil((image_height - cellSize)*1.0/stride)+1;

  //std::cout << "Feature_map_size:"<< curr_feature_map_w_ <<" "<<curr_feature_map_h_<<std::endl;
  int regOffset = curr_feature_map_w_*curr_feature_map_h_;
  // the first count numbers are confidence of face
  int count = confidence->count()/2;
  const float* confidence_data = confidence->cpu_data();
  confidence_data += count;
  const float* reg_data = reg->cpu_data();

  condidate_rects_.clear();
  for(int i=0;i<count;i++){
    if(*(confidence_data+i)>=thresh){
      int y = i / curr_feature_map_w_;
      int x = i - curr_feature_map_w_ * y;

      float xTop = (int)((x*stride+1)/scale);
      float yTop = (int)((y*stride+1)/scale);
      float xBot = (int)((x*stride+cellSize-1+1)/scale);
      float yBot = (int)((y*stride+cellSize-1+1)/scale);
      FaceRect faceRect;
      faceRect.x1 = xTop;
      faceRect.y1 = yTop;
      faceRect.x2 = xBot;
      faceRect.y2 = yBot;
      faceRect.score  = *(confidence_data+i);
      FaceInfo faceInfo;
      faceInfo.bbox = faceRect;
      faceInfo.regression = cv::Vec4f(reg_data[i+0*regOffset],reg_data[i+1*regOffset],reg_data[i+2*regOffset],reg_data[i+3*regOffset]);
      condidate_rects_.push_back(faceInfo);
    }
  }
}

MTCNN::MTCNN(const std::string &proto_model_dir){
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif
  /* Load the network. */
  PNet_.reset(new Net<float>((proto_model_dir+"/det1.prototxt"), TEST));
  PNet_->CopyTrainedLayersFrom(proto_model_dir+"/det1.caffemodel");

  CHECK_EQ(PNet_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(PNet_->num_outputs(),2) << "Network should have exactly two output, one"
                                     " is bbox and another is confidence.";

  #ifdef CPU_ONLY
  RNet_.reset(new Net<float>((proto_model_dir+"/det2.prototxt"), TEST));
  #else
  RNet_.reset(new Net<float>((proto_model_dir+"/det2_input.prototxt"), TEST));
  #endif
  RNet_->CopyTrainedLayersFrom(proto_model_dir+"/det2.caffemodel");

//  CHECK_EQ(RNet_->num_inputs(), 0) << "Network should have exactly one input.";
//  CHECK_EQ(RNet_->num_outputs(),3) << "Network should have exactly two output, one"
//                                     " is bbox and another is confidence.";

  #ifdef CPU_ONLY
  ONet_.reset(new Net<float>((proto_model_dir+"/det3.prototxt"), TEST));
  #else
  ONet_.reset(new Net<float>((proto_model_dir+"/det3_input.prototxt"), TEST));
  #endif
  ONet_->CopyTrainedLayersFrom(proto_model_dir+"/det3.caffemodel");

//  CHECK_EQ(ONet_->num_inputs(), 1) << "Network should have exactly one input.";
//  CHECK_EQ(ONet_->num_outputs(),3) << "Network should have exactly three output, one"
//                                     " is bbox and another is confidence.";

  Blob<float>* input_layer;
  input_layer = PNet_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
}

void MTCNN::WrapInputLayer(std::vector<cv::Mat>* input_channels,
        Blob<float>* input_layer, const int height, const int width) {
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void MTCNN::ClassifyFace(const std::vector<FaceInfo>& regressed_rects,cv::Mat &sample_single,
        boost::shared_ptr<Net<float> >& net,double thresh,char netName){
  int numBox = regressed_rects.size();
  Blob<float>* crop_input_layer = net->input_blobs()[0];
  int input_channels = crop_input_layer->channels();
  int input_width  = crop_input_layer->width();
  int input_height = crop_input_layer->height();
  crop_input_layer->Reshape(1, input_channels, input_width, input_height);
  net->Reshape();

  condidate_rects_.clear();
  // load crop_img data to datum
  for(int i=0;i<numBox;i++){
    std::vector<cv::Mat> channels;
    WrapInputLayer(&channels,net->input_blobs()[0],input_width,input_height);
    int pad_top   = std::abs(regressed_pading_[i].bbox.x1 - regressed_rects[i].bbox.x1);
    int pad_left  = std::abs(regressed_pading_[i].bbox.y1 - regressed_rects[i].bbox.y1);
    int pad_right = std::abs(regressed_pading_[i].bbox.y2 - regressed_rects[i].bbox.y2);
    int pad_bottom= std::abs(regressed_pading_[i].bbox.x2 - regressed_rects[i].bbox.x2);

    cv::Mat crop_img = sample_single(cv::Range(regressed_pading_[i].bbox.y1-1,regressed_pading_[i].bbox.y2),
                         cv::Range(regressed_pading_[i].bbox.x1-1,regressed_pading_[i].bbox.x2));
    cv::copyMakeBorder(crop_img,crop_img,pad_left,pad_right,pad_top,pad_bottom,cv::BORDER_CONSTANT,cv::Scalar(0));
#ifdef INTER_FAST
    cv::resize(crop_img,crop_img,cv::Size(input_width,input_height),0,0,cv::INTER_NEAREST);
#else
    cv::resize(crop_img,crop_img,cv::Size(input_width,input_height),0,0,cv::INTER_AREA);
#endif
    crop_img = (crop_img-127.5)*0.0078125;
    cv::split(crop_img,channels);

    CHECK(reinterpret_cast<float*>(channels.at(0).data) == net->input_blobs()[0]->cpu_data())
          << "Input channels are not wrapping the input layer of the network.";
    net->Forward();

    int reg_id = 0;
    int confidence_id = 1;
    if(netName == 'o') confidence_id = 2;
    const Blob<float>* reg = net->output_blobs()[reg_id];
    const Blob<float>* confidence = net->output_blobs()[confidence_id];
    // ONet points_offset != NULL
    const Blob<float>* points_offset = net->output_blobs()[1];

    const float* confidence_data = confidence->cpu_data() + confidence->count()/2;
    const float* reg_data = reg->cpu_data();
    const float* points_data;
    if(netName == 'o') points_data = points_offset->cpu_data();

    if(*(confidence_data) > thresh){
      FaceRect faceRect;
      faceRect.x1 = regressed_rects[i].bbox.x1;
      faceRect.y1 = regressed_rects[i].bbox.y1;
      faceRect.x2 = regressed_rects[i].bbox.x2;
      faceRect.y2 = regressed_rects[i].bbox.y2 ;
      faceRect.score  = *(confidence_data);
      FaceInfo faceInfo;
      faceInfo.bbox = faceRect;
      faceInfo.regression = cv::Vec4f(reg_data[0],reg_data[1],reg_data[2],reg_data[3]);

      // x x x x x y y y y y
      if(netName == 'o'){
        FacePts face_pts;
        float w = faceRect.y2 - faceRect.y1 + 1;
        float h = faceRect.x2 - faceRect.x1 + 1;
        for(int j=0;j<5;j++){
          face_pts.y[j] = faceRect.y1 + *(points_data+j) * h - 1;
          face_pts.x[j] = faceRect.x1 + *(points_data+j+5) * w -1;
        }
        faceInfo.facePts = face_pts;
      }
      condidate_rects_.push_back(faceInfo);
    }
  }
  regressed_pading_.clear();
}

// multi test image pass a forward
void MTCNN::ClassifyFace_MulImage(const std::vector<FaceInfo>& regressed_rects,cv::Mat &sample_single,
        boost::shared_ptr<Net<float> >& net,double thresh,char netName){
  condidate_rects_.clear();

  int numBox = regressed_rects.size();
  std::vector<Datum> datum_vector;

  boost::shared_ptr<MemoryDataLayer<float> > mem_data_layer;
  mem_data_layer = boost::static_pointer_cast<MemoryDataLayer<float> >(net->layers()[0]);
  int input_width  = mem_data_layer->width();
  int input_height = mem_data_layer->height();

  // load crop_img data to datum
  for(int i=0;i<numBox;i++){
    int pad_top   = std::abs(regressed_pading_[i].bbox.x1 - regressed_rects[i].bbox.x1);
    int pad_left  = std::abs(regressed_pading_[i].bbox.y1 - regressed_rects[i].bbox.y1);
    int pad_right = std::abs(regressed_pading_[i].bbox.y2 - regressed_rects[i].bbox.y2);
    int pad_bottom= std::abs(regressed_pading_[i].bbox.x2 - regressed_rects[i].bbox.x2);

    cv::Mat crop_img = sample_single(cv::Range(regressed_pading_[i].bbox.y1-1,regressed_pading_[i].bbox.y2),
                         cv::Range(regressed_pading_[i].bbox.x1-1,regressed_pading_[i].bbox.x2));
    cv::copyMakeBorder(crop_img,crop_img,pad_left,pad_right,pad_top,pad_bottom,cv::BORDER_CONSTANT,cv::Scalar(0));

#ifdef INTER_FAST
    cv::resize(crop_img,crop_img,cv::Size(input_width,input_height),0,0,cv::INTER_NEAREST);
#else
    cv::resize(crop_img,crop_img,cv::Size(input_width,input_height),0,0,cv::INTER_AREA);
#endif
    crop_img = (crop_img-127.5)*0.0078125;
    Datum datum;
    CvMatToDatumSignalChannel(crop_img,&datum);
    datum_vector.push_back(datum);
  }
  regressed_pading_.clear();

  /* extract the features and store */
  mem_data_layer->set_batch_size(numBox);
  mem_data_layer->AddDatumVector(datum_vector);
  /* fire the network */
  float no_use_loss = 0;
  net->Forward(&no_use_loss);
//  CHECK(reinterpret_cast<float*>(crop_img_set.at(0).data) == net->input_blobs()[0]->cpu_data())
//          << "Input channels are not wrapping the input layer of the network.";

  // return RNet/ONet result
  std::string outPutLayerName = (netName == 'r' ? "conv5-2" : "conv6-2");
  std::string pointsLayerName = "conv6-3";

  const boost::shared_ptr<Blob<float> > reg = net->blob_by_name(outPutLayerName);
  const boost::shared_ptr<Blob<float> > confidence = net->blob_by_name("prob1");
  // ONet points_offset != NULL
  const boost::shared_ptr<Blob<float> > points_offset = net->blob_by_name(pointsLayerName);

  const float* confidence_data = confidence->cpu_data();
  const float* reg_data = reg->cpu_data();
  const float* points_data;
  if(netName == 'o') points_data = points_offset->cpu_data();

  for(int i=0;i<numBox;i++){
    if(*(confidence_data+i*2+1) > thresh){
      FaceRect faceRect;
      faceRect.x1 = regressed_rects[i].bbox.x1;
      faceRect.y1 = regressed_rects[i].bbox.y1;
      faceRect.x2 = regressed_rects[i].bbox.x2;
      faceRect.y2 = regressed_rects[i].bbox.y2 ;
      faceRect.score  = *(confidence_data+i*2+1);
      FaceInfo faceInfo;
      faceInfo.bbox = faceRect;
      faceInfo.regression = cv::Vec4f(reg_data[4*i+0],reg_data[4*i+1],reg_data[4*i+2],reg_data[4*i+3]);

      // x x x x x y y y y y
      if(netName == 'o'){
        FacePts face_pts;
        float w = faceRect.y2 - faceRect.y1 + 1;
        float h = faceRect.x2 - faceRect.x1 + 1;
        for(int j=0;j<5;j++){
          face_pts.y[j] = faceRect.y1 + *(points_data+j+10*i) * h - 1;
          face_pts.x[j] = faceRect.x1 + *(points_data+j+5+10*i) * w -1;
        }
        faceInfo.facePts = face_pts;
      }
      condidate_rects_.push_back(faceInfo);
    }
  }
}
bool MTCNN::CvMatToDatumSignalChannel(const cv::Mat& cv_mat, Datum* datum){
  if (cv_mat.empty())
    return false;
  int channels = cv_mat.channels();

  datum->set_channels(cv_mat.channels());
  datum->set_height(cv_mat.rows);
  datum->set_width(cv_mat.cols);
  datum->set_label(0);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);

  int datum_height = datum->height();
  int datum_width  = datum->width();
  if(channels == 3){
    for(int c = 0;c < channels;c++){
      for (int h = 0; h < datum_height; ++h){
        for (int w = 0; w < datum_width; ++w){
          const float* ptr = cv_mat.ptr<float>(h);
          datum->add_float_data(ptr[w*channels+c]);
        }
      }
    }
  }

  return true;
}

void MTCNN::Detect(const cv::Mat& image,std::vector<FaceInfo>& faceInfo,int minSize,double* threshold,double factor){

  // 2~3ms
  // invert to RGB color space and float type
  cv::Mat sample_single,resized;
  image.convertTo(sample_single,CV_32FC3);
  cv::cvtColor(sample_single,sample_single,cv::COLOR_BGR2RGB);
  sample_single = sample_single.t();

  int height = image.rows;
  int width  = image.cols;
  int minWH = std::min(height,width);
  int factor_count = 0;
  double m = 12./minSize;
  minWH *= m;
  std::vector<double> scales;
  while (minWH >= 12)
  {
    scales.push_back(m * std::pow(factor,factor_count));
    minWH *= factor;
    ++factor_count;
  }

  // 11ms main consum
  Blob<float>* input_layer = PNet_->input_blobs()[0];
  for(int i=0;i<factor_count;i++)
  {
    double scale = scales[i];
    int ws = std::ceil(height*scale);
    int hs = std::ceil(width*scale);

    // wrap image and normalization using INTER_AREA method
    cv::resize(sample_single,resized,cv::Size(ws,hs),0,0,cv::INTER_AREA);
#ifdef INTER_FAST
    cv::resize(sample_single,resized,cv::Size(ws,hs),0,0,cv::INTER_NEAREST);
#else
    cv::resize(sample_single,resized,cv::Size(ws,hs),0,0,cv::INTER_AREA);
#endif
    resized.convertTo(resized, CV_32FC3, 0.0078125,-127.5*0.0078125);

    // input data
    input_layer->Reshape(1, 3, hs, ws);
    PNet_->Reshape();
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels,PNet_->input_blobs()[0],hs,ws);
    cv::split(resized,input_channels);

    // check data transform right
    CHECK(reinterpret_cast<float*>(input_channels.at(0).data) == PNet_->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
    PNet_->Forward();

    // return result
    Blob<float>* reg = PNet_->output_blobs()[0];
    //const float* reg_data = reg->cpu_data();
    Blob<float>* confidence = PNet_->output_blobs()[1];
    GenerateBoundingBox(confidence, reg, scale, threshold[0],ws,hs);
    std::vector<FaceInfo> bboxes_nms = NonMaximumSuppression(condidate_rects_,0.5,'u');
    total_boxes_.insert(total_boxes_.end(),bboxes_nms.begin(),bboxes_nms.end());
  }

  int numBox = total_boxes_.size();
  if(numBox != 0){
    total_boxes_ = NonMaximumSuppression(total_boxes_,0.7,'u');
    regressed_rects_ = BoxRegress(total_boxes_,1);
    total_boxes_.clear();

    Bbox2Square(regressed_rects_);
    Padding(width,height);

    /// Second stage
    #ifdef CPU_ONLY
    ClassifyFace(regressed_rects_,sample_single,RNet_,threshold[1],'r');
    #else
    ClassifyFace_MulImage(regressed_rects_,sample_single,RNet_,threshold[1],'r');
    #endif
    condidate_rects_ = NonMaximumSuppression(condidate_rects_,0.7,'u');
    regressed_rects_ = BoxRegress(condidate_rects_,2);

    Bbox2Square(regressed_rects_);
    Padding(width,height);

    /// three stage
    numBox = regressed_rects_.size();
    if(numBox != 0){
      #ifdef CPU_ONLY
      ClassifyFace(regressed_rects_,sample_single,ONet_,threshold[2],'o');
      #else
      ClassifyFace_MulImage(regressed_rects_,sample_single,ONet_,threshold[2],'o');
      #endif
      regressed_rects_ = BoxRegress(condidate_rects_,3);
      faceInfo = NonMaximumSuppression(regressed_rects_,0.7,'m');
    }
  }
  regressed_pading_.clear();
  regressed_rects_.clear();
  condidate_rects_.clear();
}

int main7(int argc,char **argv)
{
  if(argc != 3){
    std::cout << "MTMain.bin [model dir] [imagePath]"<<std::endl;
    return 0;
  }
  ::google::InitGoogleLogging(argv[0]);
  double threshold[3] = {0.6,0.7,0.7};
  double factor = 0.709;
  int minSize = 40;
  std::string proto_model_dir = argv[1];
  MTCNN detector(proto_model_dir);





  std::string imageName = argv[2];

  cv::Mat image = cv::imread(imageName);
  std::vector<FaceInfo> faceInfo;
  clock_t t1 = clock();
  std::cout <<"Detect "<<image.rows<<"X"<<image.cols;
  detector.Detect(image,faceInfo,minSize,threshold,factor);
  std::cout<<CLOCKS_PER_SEC<<std::endl;
  #ifdef CPU_ONLY
  std::cout<<" Time Using CPU: " << (clock() - t1)*1.0/CLOCKS_PER_SEC<<"s"<<std::endl;
  #else
  std::cout<<" Time Using GPU-CUDNN: " << (clock() - t1)*1.0/CLOCKS_PER_SEC<<"s"<<std::endl;
  #endif
  for(int i = 0;i<faceInfo.size();i++){
    float x = faceInfo[i].bbox.x1;
    float y = faceInfo[i].bbox.y1;
    float h = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 +1;
    float w = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 +1;
    std::cout<<x<<":"<<y<<":"<<w<<":"<<h<<std::endl;
    cv::rectangle(image,cv::Rect(y,x,w,h),cv::Scalar(255,0,0),2);
  }
  for(int i=0;i<faceInfo.size();i++){
    FacePts facePts = faceInfo[i].facePts;
    for(int j=0;j<5;j++)
      cv::circle(image,cv::Point(facePts.y[j],facePts.x[j]),1,cv::Scalar(255,255,0),2);
  }
  cv::imshow("a",image);
  cv::waitKey(0);

  return 1;
}
/**
 *
 *
 * fddb test
**/

int main4(int argc,char **argv)
{
  if(argc != 3){
    std::cout << "MTMain.bin [model dir] [imagePath]"<<std::endl;
    return 0;
  }
  ::google::InitGoogleLogging(argv[0]);
  double threshold[3] = {0.6,0.7,0.7};
  double factor = 0.709;
  int minSize = 40;
  std::string proto_model_dir = argv[1];
  MTCNN detector(proto_model_dir);



  string file="/home/pub/samba/fddb/FDDB-folds/Fold_all.txt";
  std::ofstream result2("result.txt",ios::out);
  std::ifstream in;
  string line;
  in.open(file.c_str(),ios::in);
  while(getline(in,line)){
			string pic_road="/home/pub/samba/fddb/"+line+".jpg";
			std::cout<<pic_road<<std::endl;


  std::string imageName = pic_road;

  cv::Mat image = cv::imread(imageName);
  std::vector<FaceInfo> faceInfo;
  clock_t t1 = clock();
  std::cout <<"Detect "<<image.rows<<"X"<<image.cols;
  Caffe::set_mode(Caffe::GPU);
  detector.Detect(image,faceInfo,minSize,threshold,factor);
  std::cout<<CLOCKS_PER_SEC<<std::endl;
  #ifdef CPU_ONLY
  std::cout<<" Time Using CPU: " << (clock() - t1)*1.0/CLOCKS_PER_SEC<<"s"<<std::endl;
  #else
  std::cout<<" Time Using GPU-CUDNN: " << (clock() - t1)*1.0/CLOCKS_PER_SEC<<"s"<<std::endl;
  #endif
result2<<line<<std::endl<<faceInfo.size()<<std::endl;
  for(int i = 0;i<faceInfo.size();i++){
    float x = faceInfo[i].bbox.x1;
    float y = faceInfo[i].bbox.y1;
    float h = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 +1;
    float w = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 +1;
    float score=faceInfo[i].bbox.score;
    //cv::rectangle(image,cv::Rect(y,x,w,h),cv::Scalar(255,0,0),2);
    result2<<y<<" "<<x<<" "<<w<<" "<<h<<" "<<score<<std::endl;;
    result2.flush();
  }
  }
	result2.close();

  return 1;
}


std::vector<std::string> split(const  std::string& s, const std::string& delim)
{
	std::vector<std::string> elems;
	size_t pos = 0;
	size_t len = s.length();
	size_t delim_len = delim.length();
	if (delim_len == 0) return elems;
	while (pos < len)
	{
		int find_pos = s.find(delim, pos);
		if (find_pos < 0)
		{
			elems.push_back(s.substr(pos, len - pos));
			break;
		}
		elems.push_back(s.substr(pos, find_pos - pos));
		pos = find_pos + delim_len;
	}
	return elems;
}
/**
 * lfw face get
 */
int main22(int argc,char **argv)
{
  if(argc != 3){
    std::cout << "MTMain.bin [model dir] [imagePath]"<<std::endl;
    return 0;
  }
  ::google::InitGoogleLogging(argv[0]);
  double threshold[3] = {0.6,0.7,0.7};
  double factor = 0.709;
  int minSize = 40;
  std::string proto_model_dir = argv[1];
  MTCNN detector(proto_model_dir);



  string file="/home/pub/Work/caffe-lfw-roc/lfw_list2.txt";
  std::ifstream in;
  string line;
  in.open(file.c_str(),ios::in);
  std::ofstream result2("no.txt",ios::out);
  while(getline(in,line)){
			string pic_road="/home/pub/samba/lfw_funneled/"+split(line," ")[0];
		//	string pic_save="/home/pub/samba/lfw_funneled_onlyface/"+split(line," ")[0];
			string pic_save="/home/pub/samba/lfw_funneled_samecrop/"+split(line," ")[0];
			//pic_road="/home/pub/samba/lfw_funneled/Doug_Collins/Doug_Collins_0002.jpg";
			//pic_save="/home/pub/samba/lfw_funneled_onlyface/Doug_Collins/Doug_Collins_0002.jpg";
			string name=split(split(line," ")[0],"/")[1];
			std::cout<<pic_road<<std::endl;
			std::cout<<name<<std::endl;


  std::string imageName = pic_road;

  cv::Mat image = cv::imread(imageName);
  std::vector<FaceInfo> faceInfo;
  clock_t t1 = clock();
  std::cout <<"Detect "<<image.rows<<"X"<<image.cols;
  Caffe::set_mode(Caffe::GPU);
  //detector.Detect(image,faceInfo,minSize,threshold,factor);
  std::cout<<CLOCKS_PER_SEC<<std::endl;
  #ifdef CPU_ONLY
  std::cout<<" Time Using CPU: " << (clock() - t1)*1.0/CLOCKS_PER_SEC<<"s"<<std::endl;
  #else
  std::cout<<" Time Using GPU-CUDNN: " << (clock() - t1)*1.0/CLOCKS_PER_SEC<<"s"<<std::endl;
  #endif

#if 1
  // use same crop
    cv::Mat faceROI = image(cv::Rect(78,70,94,125));  
    imwrite(pic_save, faceROI);  
    continue;
#endif
  if(faceInfo.size()>1){
    cv::Mat faceROI = image(cv::Rect(78,70,94,125));  
    imwrite(pic_save, faceROI);  
    result2<<line<<std::endl;
    continue;
  }
  for(int i = 0;i<faceInfo.size();i++){
    float x = faceInfo[i].bbox.x1;
    float y = faceInfo[i].bbox.y1;
    float h = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 +1;
    float w = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 +1;
    if(x+h>image.rows)
	    h=image.rows-x-1;
    if(y+w>image.cols)
	    w=image.cols-y-1;
    float score=faceInfo[i].bbox.score;
    std::cout<<y<<":"<<x<<":"<<w<<":"<<h<<std::endl;
    //cv::rectangle(image,cv::Rect(y,x,w,h),cv::Scalar(255,0,0),2);
    cv::Mat faceROI = image(cv::Rect(y,x,w,h));  
    imwrite(pic_save, faceROI);  
  }
  }
  result2.close();

  return 1;
}
/**
 * web face get
 */
int main88(int argc,char **argv)
{
  if(argc != 3){
    std::cout << "MTMain.bin [model dir] [imagePath]"<<std::endl;
    return 0;
  }
  ::google::InitGoogleLogging(argv[0]);
  double threshold[3] = {0.6,0.7,0.7};
  double factor = 0.709;
  int minSize = 40;
  std::string proto_model_dir = argv[1];
  MTCNN detector(proto_model_dir);



  string file="/home/pub/Work/YOLO/darknet/not.txt";
  std::ifstream in;
  string line;
  in.open(file.c_str(),ios::in);
  std::ofstream result2("no2.txt",ios::out);
  while(getline(in,line)){
			string pic_road="/home/pub/samba/CASIA-WebFace/align/"+line;
			string name=split(line,"/")[1];
			std::cout<<pic_road<<std::endl;
			//std::cout<<name<<std::endl;
			string pic_save="/home/pub/samba/CASIA-WebFace/align/"+line;


  std::string imageName = pic_road;

  cv::Mat image = cv::imread(imageName);
  std::vector<FaceInfo> faceInfo;
  clock_t t1 = clock();
  std::cout <<"Detect "<<image.rows<<"X"<<image.cols;
  Caffe::set_mode(Caffe::GPU);
  detector.Detect(image,faceInfo,minSize,threshold,factor);
  std::cout<<CLOCKS_PER_SEC<<std::endl;
  #ifdef CPU_ONLY
  std::cout<<" Time Using CPU: " << (clock() - t1)*1.0/CLOCKS_PER_SEC<<"s"<<std::endl;
  #else
  std::cout<<" Time Using GPU-CUDNN: " << (clock() - t1)*1.0/CLOCKS_PER_SEC<<"s"<<std::endl;
  #endif

#if 0
  // use same crop
    cv::Mat faceROI = image(cv::Rect(78,70,94,125));  
    imwrite(pic_save, faceROI);  
    continue;
#endif
  if(faceInfo.size()>1){
    imwrite(pic_save, image);  
    result2<<line<<std::endl;
    continue;
  }
  for(int i = 0;i<faceInfo.size();i++){
    float x = faceInfo[i].bbox.x1;
    float y = faceInfo[i].bbox.y1;
    float h = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 +1;
    float w = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 +1;
    if(x+h>image.rows)
	    h=image.rows-x-1;
    if(y+w>image.cols)
	    w=image.cols-y-1;
    if(h/w>2.2 || x <0 || y <0 ){
	    imwrite(pic_save, image);  
	    result2<<line<<std::endl;
	    break;

    }
    float score=faceInfo[i].bbox.score;
    std::cout<<y<<":"<<x<<":"<<w<<":"<<h<<std::endl;
    //cv::rectangle(image,cv::Rect(y,x,w,h),cv::Scalar(255,0,0),2);
    cv::Mat faceROI = image(cv::Rect(y,x,w,h));  
    imwrite(pic_save, faceROI);  
  }

  }
  result2.close();

  return 1;
}
bool SortByM1( const Mat &v1, const Mat&v2)  
{  
	    return v1.cols*v1.rows > v2.cols*v2.rows;  
}  
vector<Mat> MTCNN::getFaceRects(const Mat& img,vector<Rect>* faceRects){       //给定一张图片，将其中人脸ROI返回,以及对应的五个特征点
        double threshold[3] = {0.6,0.7,0.7};
        double factor = 0.709;
        int minSize = 40;
	if(faceRects!=NULL)
		faceRects->clear();
	vector<FaceInfo> faceInfo;
	this->Detect(img, faceInfo,minSize,threshold,factor);   //得到人脸框
	int n = faceInfo.size();      //人脸个数

	vector<Mat> faces;      //n个人脸ROI
	for (int i = 0; i<n; ++i){           //对每个人脸区域
		float x1 = std::max<float>(0, faceInfo[i].bbox.y1 - 20);
		float y1 = std::max<float>(0, faceInfo[i].bbox.x1 - 20);
		float x2 = std::min<float>(img.cols - 1, faceInfo[i].bbox.y2 + 20);
		float y2 = std::min<float>(img.rows - 1, faceInfo[i].bbox.x2 + 20);
		//printf("%f %f %f %f\n",x1,y1,x2,y2);

		if(y2/x2>2.2 ){
			printf("asdfaf %d\n",n);
			if (i<n){
			printf("asdfaf %d\n",i);
				continue;
			}
		}
		printf("dd %d\n",i);


		//printf("%.f  %.f  %.f  %.f\n",x1,y1,x2,y2);
		Mat faceROI = img(Rect(Point(x1, y1), Point(x2, y2))).clone();   //人脸ROI

		//保存检测到的人脸框
		if(faceRects!=NULL)
			faceRects->push_back(Rect(Point(x1, y1), Point(x2, y2))); 

		FacePts pts = faceInfo[i].facePts;   //五个关键点
		Point2f lefteye(pts.y[0], pts.x[0]);
		Point2f righteye(pts.y[1], pts.x[1]);
		Point2f center = Point2f((lefteye.x + righteye.x)*0.5, (lefteye.y + righteye.y)*0.5);  //两眼的中心点
		double dy = righteye.y - lefteye.y;
		double dx = righteye.x - lefteye.x;
		double angle = atan2(dy, dx)*180.0 / CV_PI;     //角度

		Mat rot_mat = getRotationMatrix2D(center, angle, 1.0);   //求得仿射矩阵
		Mat rot;         //变换后的人脸图像
		warpAffine(faceROI, rot, rot_mat, faceROI.size());    //仿射变换
		//仿射变换后，需要再定位一次人脸区域
		vector<FaceInfo> tmp;
		this->Detect(rot, tmp,minSize,threshold,factor);
		if (tmp.size() == 0){    //变换后没有检测到人脸,则保留变换前的人脸ROI
			faces.push_back(faceROI.clone());
			//imshow("err",faces[i]);
			continue;
		}
		//否则，将新的人脸返回
		x1 = std::max<float>(0, tmp[0].bbox.y1);
		y1 = std::max<float>(0, tmp[0].bbox.x1);
		x2 = std::min<float>(rot.cols - 1, tmp[0].bbox.y2);
		y2 = std::min<float>(rot.rows - 1, tmp[0].bbox.x2);
		faces.push_back( rot(Rect(Point(x1, y1), Point(x2, y2))).clone());
	}

	if(faces.size()>1)
		std::sort(faces.begin(),faces.end(),SortByM1);  
	return faces;
}
/**
 * web face get2
 */
int main(int argc,char **argv)
{
  if(argc != 3){
    std::cout << "MTMain.bin [model dir] [imagePath]"<<std::endl;
    return 0;
  }
  ::google::InitGoogleLogging(argv[0]);
  double threshold[3] = {0.6,0.7,0.7};
  double factor = 0.709;
  int minSize = 40;
  std::string proto_model_dir = argv[1];
  MTCNN detector(proto_model_dir);



  string file="/home/pub/samba/CASIA-WebFace/w2.txt";
  std::ifstream in;
  string line;
  in.open(file.c_str(),ios::in);
  std::ofstream result2("no2.txt",ios::out);
  Caffe::set_mode(Caffe::GPU);
 int  countt=0;


  while(getline(in,line)){
  //			clock_t t1 = clock();
	  		printf("%d\n",countt++);
			line=split(line," ")[0];
			string pic_road="/home/pub/samba/CASIA-WebFace/CASIA-WebFace/"+line;
			std::cout<<pic_road<<std::endl;
			//std::cout<<name<<std::endl;
			string pic_save="/home/pub/samba/CASIA-WebFace/align2/"+line;


  std::string imageName = pic_road;

  cv::Mat image = cv::imread(imageName);

  std::vector<Rect> faces;
  vector<Mat> ffs= detector.getFaceRects(image,&faces);
  if(faces.size()==0){
    cv::resize(image, image, cv::Size(144, 144), (0, 0), (0, 0), cv::INTER_LINEAR);
    imwrite(pic_save, image);  
    result2<<line<<std::endl;
    continue;
  }else{
    cv::resize(ffs[0], ffs[0], cv::Size(144, 144), (0, 0), (0, 0), cv::INTER_LINEAR);
    imwrite(pic_save, ffs[0]);  
  }
  //std::cout<<" Time Using CPU: " << (clock() - t1)*1.0/CLOCKS_PER_SEC<<"s"<<std::endl;
  }
  result2.close();

  return 1;
}
