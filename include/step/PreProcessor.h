#include "../Configuration.h"
#include "../DataTypes.h"
using config::CameraParameters;
using config::GlobalConfiguration;

//预处理器
//负责完成图片类型的转换，金字塔生成，以及双边滤波
namespace step{

    class PreProcessor{
    public:
      PreProcessor(CameraParameters cameraParameters,GlobalConfiguration globalConfiguration);
      //预处理
      FrameData preProcess(const cv::Mat& depth_map);
    private:
      CameraParameters cameraParameters;
      GlobalConfiguration globalConfiguration;
    };
    
}