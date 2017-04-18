#include "FilmGradePlugin.h"

#include <stdio.h>
#include <cmath>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"

#define kPluginName "FilmGrade"
#define kPluginGrouping "OpenFX Yo"
#define kPluginDescription "Film style grading"
#define kPluginIdentifier "OpenFX.Yo.FilmGrade"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 2

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

////////////////////////////////////////////////////////////////////////////////

namespace {
    struct RGBValues {
        double r,g,b;
        RGBValues(double v) : r(v), g(v), b(v) {}
        RGBValues() : r(0), g(0), b(0) {}
    };
}


class ImageScaler : public OFX::ImageProcessor
{
public:
    explicit ImageScaler(OFX::ImageEffect& p_Instance);

	virtual void processImagesCUDA();
    virtual void processImagesOpenCL();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

    void setSrcImg(OFX::Image* p_SrcImg);
    void setScales(float p_ExpR, float p_ExpG, float p_ExpB, 
    float p_ContR, float p_ContG, float p_ContB, float p_SatR, float p_SatG, float p_SatB, 
    float p_ShadR, float p_ShadG, float p_ShadB, float p_MidR, float p_MidG, float p_MidB, 
    float p_HighR, float p_HighG, float p_HighB, float p_ShadP, float p_HighP, float p_ContP, float p_Display);

private:
    OFX::Image* _srcImg;
    float _exp[3];
    float _cont[3];
    float _sat[3];
    float _shad[3];
    float _mid[3];
    float _high[3];
    float _pivot[3];
    float _display[1];
    
};

ImageScaler::ImageScaler(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}

extern void RunCudaKernel(int p_Width, int p_Height, float* p_Exp, float* p_Cont, float* p_Sat, 
float* p_Shad, float* p_Mid, float* p_High, float* p_Pivot, float* p_Display, const float* p_Input, float* p_Output);

void ImageScaler::processImagesCUDA()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunCudaKernel(width, height, _exp, _cont, _sat, _shad, _mid, _high, _pivot, _display, input, output);
}

extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_Exp, float* p_Cont, float* p_Sat, 
float* p_Shad, float* p_Mid, float* p_High, float* p_Pivot, float* p_Display, const float* p_Input, float* p_Output);

void ImageScaler::processImagesOpenCL()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunOpenCLKernel(_pOpenCLCmdQ, width, height, _exp, _cont, _sat, _shad, _mid, _high, _pivot, _display, input, output);
}

void ImageScaler::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
    for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
    {
        if (_effect.abort()) break;

        float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));

        for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
        {
            float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);

            // do we have a source image to scale up
            if (srcPix)
            {
            	  float width = p_ProcWindow.x2;
            	  float height = p_ProcWindow.y2;
            	  
            	  float e = 2.718281828459045;
            	  float pie = 3.141592653589793;
            	  
            	  float Red = _display[0] != 1.0f ? srcPix[0] : (x / width);
            	  float Green = _display[0] != 1.0f ? srcPix[1] : (x / width);
            	  float Blue = _display[0] != 1.0f ? srcPix[2] : (x / width);
                  
                  float expr1 = (_pivot[0] / 2.0f) - (1.0f - _pivot[1])/4.0f;
				  float expr2 = (1.0f - (1.0f - _pivot[1])/2.0f) + (_pivot[0] / 4.0f);
				  float expr3R = (Red - expr1) / (expr2 - expr1);
				  float expr3G = (Green - expr1) / (expr2 - expr1);
				  float expr3B = (Blue - expr1) / (expr2 - expr1);
				  float expr4 =  _pivot[2] < 0.5f ? 0.5f - (0.5f - _pivot[2])/2.0f : 0.5f + (_pivot[2] - 0.5f)/2.0f;
				  float expr5R = expr3R > expr4 ? (expr3R - expr4) / (2.0f - 2.0f*expr4) + 0.5f : expr3R /(2.0f*expr4);
				  float expr5G = expr3G > expr4 ? (expr3G - expr4) / (2.0f - 2.0f*expr4) + 0.5f : expr3G /(2.0f*expr4);
				  float expr5B = expr3B > expr4 ? (expr3B - expr4) / (2.0f - 2.0f*expr4) + 0.5f : expr3B /(2.0f*expr4);
				  float expr6R = (((sin(2.0f * pie * (expr5R -1.0f/4.0f)) + 1.0f) / 20.0f) * _mid[0]*4.0f) + expr3R;
				  float expr6G = (((sin(2.0f * pie * (expr5G -1.0f/4.0f)) + 1.0f) / 20.0f) * _mid[1]*4.0f) + expr3G;
				  float expr6B = (((sin(2.0f * pie * (expr5B -1.0f/4.0f)) + 1.0f) / 20.0f) * _mid[2]*4.0f) + expr3B;
				  float midR = Red >= expr1 && Red <= expr2 ? expr6R * (expr2 - expr1) + expr1 : Red;
				  float midG = Green >= expr1 && Green <= expr2 ? expr6G * (expr2 - expr1) + expr1 : Green;
				  float midB = Blue >= expr1 && Blue <= expr2 ? expr6B * (expr2 - expr1) + expr1 : Blue;

                  float shadupR1 = 2.0f * (midR/_pivot[0]) - log((midR/_pivot[0]) * (e * _shad[0] * 2.0f) + 1.0f)/log(e * _shad[0] * 2.0f + 1.0f);
                  float shadupR = midR < _pivot[0] && _shad[0] > 0.0f ? (shadupR1 + _shad[0] * (1.0f - shadupR1)) * _pivot[0] : midR;
                  float shadupG1 = 2.0f * (midG/_pivot[0]) - log((midG/_pivot[0]) * (e * _shad[1] * 2.0f) + 1.0f)/log(e * _shad[1] * 2.0f + 1.0f);
                  float shadupG = midG < _pivot[0] && _shad[1] > 0.0f ? (shadupG1 + _shad[1] * (1.0f - shadupG1)) * _pivot[0] : midG;
                  float shadupB1 = 2.0f * (midB/_pivot[0]) - log((midB/_pivot[0]) * (e * _shad[2] * 2.0f) + 1.0f)/log(e * _shad[2] * 2.0f + 1.0f);
                  float shadupB = midB < _pivot[0] && _shad[2] > 0.0f ? (shadupB1 + _shad[2] * (1.0f - shadupB1)) * _pivot[0] : midB;
                  
                  /*
                  float shaddownR1 = log((shadupR/_pivot[0]) * (e * _shad[0] * -10.0f) + 1.0f)/log(e * _shad[0] * -10.0f + 1.0f);
                  float shaddownR = shadupR < _pivot[0] && _shad[0] < 0.0f ? (shaddownR1 + _shad[0] * 10.0f * (1.0f - shaddownR1)) * _pivot[0] : shadupR;
                  float shaddownG1 = log((shadupG/_pivot[0]) * (e * _shad[0] * -10.0f) + 1.0f)/log(e * _shad[0] * -10.0f + 1.0f);
                  float shaddownG = shadupG < _pivot[0] && _shad[0] < 0.0f ? (shaddownG1 + _shad[0] * 10.0f * (1.0f - shaddownG1)) * _pivot[0] : shadupG;
                  float shaddownB1 = log((shadupB/_pivot[0]) * (e * _shad[0] * -10.0f) + 1.0f)/log(e * _shad[0] * -10.0f + 1.0f);
                  float shaddownB = shadupB < _pivot[0] && _shad[0] < 0.0f ? (shaddownB1 + _shad[0] * 10.0f * (1.0f - shaddownB1)) * _pivot[0] : shadupB;
                  */
                  float shaddownR1 = (shadupR/_pivot[0]) + (_shad[0] * 2.0f * (1.0f - shadupR/_pivot[0]));
                  float shaddownR = shadupR < _pivot[0] && _shad[0] < 0.0f ? (shaddownR1 >= 0.0f ? log(shaddownR1 * (e * _shad[0] * -2.0f) + 1.0f)/log(e * _shad[0] * -2.0f + 1.0f) : shaddownR1) * _pivot[0] : shadupR;
                  float shaddownG1 = (shadupG/_pivot[0]) + (_shad[1] * 2.0f * (1.0f - shadupG/_pivot[0]));
                  float shaddownG = shadupG < _pivot[0] && _shad[1] < 0.0f ? (shaddownG1 >= 0.0f ? log(shaddownG1 * (e * _shad[1] * -2.0f) + 1.0f)/log(e * _shad[1] * -2.0f + 1.0f) : shaddownG1) * _pivot[0] : shadupG;
                  float shaddownB1 = (shadupB/_pivot[0]) + (_shad[2] * 2.0f * (1.0f - shadupB/_pivot[0]));
                  float shaddownB = shadupB < _pivot[0] && _shad[2] < 0.0f ? (shaddownB1 >= 0.0f ? log(shaddownB1 * (e * _shad[2] * -2.0f) + 1.0f)/log(e * _shad[2] * -2.0f + 1.0f) : shaddownB1) * _pivot[0] : shadupB;
                  
                  float highupR1 = ((shaddownR - _pivot[1]) / (1.0f - _pivot[1])) * (1.0f + (_high[0] * 2.0f));
                  float highupR = shaddownR > _pivot[1] && _pivot[1] < 1.0f && _high[0] > 0.0f ? (2.0f * highupR1 - log(highupR1 * e * _high[0] + 1.0f)/log(e * _high[0] + 1.0f)) * (1.0f - _pivot[1]) + _pivot[1] : shaddownR;
                  float highupG1 = ((shaddownG - _pivot[1]) / (1.0f - _pivot[1])) * (1.0f + (_high[1] * 2.0f));
                  float highupG = shaddownG > _pivot[1] && _pivot[1] < 1.0f && _high[1] > 0.0f ? (2.0f * highupG1 - log(highupG1 * e * _high[1] + 1.0f)/log(e * _high[1] + 1.0f)) * (1.0f - _pivot[1]) + _pivot[1] : shaddownG;
                  float highupB1 = ((shaddownB - _pivot[1]) / (1.0f - _pivot[1])) * (1.0f + (_high[2] * 2.0f));
                  float highupB = shaddownB > _pivot[1] && _pivot[1] < 1.0f && _high[2] > 0.0f ? (2.0f * highupB1 - log(highupB1 * e * _high[2] + 1.0f)/log(e * _high[2] + 1.0f)) * (1.0f - _pivot[1]) + _pivot[1] : shaddownB;
                  
                  float highdownR1 = (highupR - _pivot[1]) / (1.0f - _pivot[1]);
                  float highdownR = highupR > _pivot[1] && _pivot[1] < 1.0f && _high[0] < 0.0f ? log(highdownR1 * (e * _high[0] * -2.0f) + 1.0f)/log(e * _high[0] * -2.0f + 1.0f) * (1.0f + _high[0]) * (1.0f - _pivot[1]) + _pivot[1]  : highupR;
                  float highdownG1 = (highupG - _pivot[1]) / (1.0f - _pivot[1]);
                  float highdownG = highupG > _pivot[1] && _pivot[1] < 1.0f && _high[1] < 0.0f ? log(highdownG1 * (e * _high[1] * -2.0f) + 1.0f)/log(e * _high[1] * -2.0f + 1.0f) * (1.0f + _high[1]) * (1.0f - _pivot[1]) + _pivot[1]  : highupG;
                  float highdownB1 = (highupB - _pivot[1]) / (1.0f - _pivot[1]);
                  float highdownB = highupB > _pivot[1] && _pivot[1] < 1.0f && _high[2] < 0.0f ? log(highdownB1 * (e * _high[2] * -2.0f) + 1.0f)/log(e * _high[2] * -2.0f + 1.0f) * (1.0f + _high[2]) * (1.0f - _pivot[1]) + _pivot[1]  : highupB;
                  
                  float expR = highdownR + _exp[0]/100.0f;
                  float expG = highdownG + _exp[1]/100.0f;
                  float expB = highdownB + _exp[2]/100.0f;
                  
                  float contR = (expR - _pivot[2]) * _cont[0] + _pivot[2];
                  float contG = (expG - _pivot[2]) * _cont[1] + _pivot[2];
                  float contB = (expB - _pivot[2]) * _cont[2] + _pivot[2];
                  
                  float luma = contR * 0.2126f + contG * 0.7152f + contB * 0.0722f;
                  float satR = (1.0f - (_sat[0]*0.2126f + _sat[1]* 0.7152f + _sat[2]*0.0722f)) * luma + contR * _sat[0];
                  float satG = (1.0f - (_sat[0]*0.2126f + _sat[1]* 0.7152f + _sat[2]*0.0722f)) * luma + contG * _sat[1];
                  float satB = (1.0f - (_sat[0]*0.2126f + _sat[1]* 0.7152f + _sat[2]*0.0722f)) * luma + contB * _sat[2];
                  
                  float outR = _display[0] != 1.0f ? satR : y/(height) >= _pivot[0] && y/(height) <= _pivot[0] + 0.005f ? (fmod(x, 2.0f) != 0.0f ? 1.0f : 0.0f) : satR >= (y - 5)/(height) && satR <= (y + 5)/(height) ? 1.0f : 0.0f;
                  float outG = _display[0] != 1.0f ? satG : y/(height) >= _pivot[1] && y/(height) <= _pivot[1] + 0.005f ? (fmod(x, 2.0f) != 0.0f ? 1.0f : 0.0f) : satG >= (y - 5)/(height) && satG <= (y + 5)/(height) ? 1.0f : 0.0f;
                  float outB = _display[0] != 1.0f ? satB : y/(height) >= _pivot[2] && y/(height) <= _pivot[2] + 0.005f ? (fmod(x, 2.0f) != 0.0f ? 1.0f : 0.0f) : satB >= (y - 5)/(height) && satB <= (y + 5)/(height) ? 1.0f : 0.0f;
                            
                  dstPix[0] = outR;
                  dstPix[1] = outG;
                  dstPix[2] = outB;
                  dstPix[3] = srcPix[3];
                    
            }
            else
            {
                // no src pixel here, be black and transparent
                for (int c = 0; c < 4; ++c)
                {
                    dstPix[c] = 0;
                }
            }

            // increment the dst pixel
            dstPix += 4;
        }
    }
}

void ImageScaler::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

void ImageScaler::setScales(float p_ExpR, float p_ExpG, float p_ExpB, 
float p_ContR, float p_ContG, float p_ContB, float p_SatR, float p_SatG, float p_SatB, 
float p_ShadR, float p_ShadG, float p_ShadB, float p_MidR, float p_MidG, float p_MidB, 
float p_HighR, float p_HighG, float p_HighB, float p_ShadP, float p_HighP, float p_ContP, float p_Display)
{
    _exp[0] = p_ExpR;
    _exp[1] = p_ExpG;
    _exp[2] = p_ExpB;
    _cont[0] = p_ContR;
    _cont[1] = p_ContG;
    _cont[2] = p_ContB;
    _sat[0] = p_SatR;
    _sat[1] = p_SatG;
    _sat[2] = p_SatB;
    _shad[0] = p_ShadR;
    _shad[1] = p_ShadG;
    _shad[2] = p_ShadB;
    _mid[0] = p_MidR;
    _mid[1] = p_MidG;
    _mid[2] = p_MidB;
    _high[0] = p_HighR;
    _high[1] = p_HighG;
    _high[2] = p_HighB;
    _pivot[0] = p_ShadP;
    _pivot[1] = p_HighP;
    _pivot[2] = p_ContP;
    _display[0] = p_Display;
}

////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class FilmGradePlugin : public OFX::ImageEffect
{
public:
    explicit FilmGradePlugin(OfxImageEffectHandle p_Handle);

    /* Override the render */
    virtual void render(const OFX::RenderArguments& p_Args);

    /* Override is identity */
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);
    
    /* Override changedParam */
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

    /* Set up and run a processor */
    void setupAndProcess(ImageScaler &p_ImageScaler, const OFX::RenderArguments& p_Args);

private:
    // Does not own the following pointers
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;

	OFX::RGBParam *m_ExpSwatch;
	OFX::RGBParam *m_ContSwatch;
	OFX::RGBParam *m_SatSwatch;
	OFX::RGBParam *m_ShadSwatch;
	OFX::RGBParam *m_MidSwatch;
	OFX::RGBParam *m_HighSwatch;
	
	OFX::DoubleParam* m_Exp;
	OFX::DoubleParam* m_ExpR;
	OFX::DoubleParam* m_ExpG;
	OFX::DoubleParam* m_ExpB;
	OFX::DoubleParam* m_Cont;
	OFX::DoubleParam* m_ContR;
	OFX::DoubleParam* m_ContG;
	OFX::DoubleParam* m_ContB;
	OFX::DoubleParam* m_Sat;
	OFX::DoubleParam* m_SatR;
	OFX::DoubleParam* m_SatG;
	OFX::DoubleParam* m_SatB;
	OFX::DoubleParam* m_Shad;
	OFX::DoubleParam* m_ShadR;
	OFX::DoubleParam* m_ShadG;
	OFX::DoubleParam* m_ShadB;
	OFX::DoubleParam* m_Mid;
	OFX::DoubleParam* m_MidR;
	OFX::DoubleParam* m_MidG;
	OFX::DoubleParam* m_MidB;
	OFX::DoubleParam* m_High;
	OFX::DoubleParam* m_HighR;
	OFX::DoubleParam* m_HighG;
	OFX::DoubleParam* m_HighB;
	OFX::DoubleParam* m_ShadP;
	OFX::DoubleParam* m_ShadPP;
	OFX::DoubleParam* m_HighP;
	OFX::DoubleParam* m_HighPP;
	OFX::DoubleParam* m_ContP;
	OFX::DoubleParam* m_ContPP;
	OFX::BooleanParam* m_Display;
	
};

FilmGradePlugin::FilmGradePlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

	m_ExpSwatch = fetchRGBParam("expSwatch");
	m_ContSwatch = fetchRGBParam("contSwatch");
	m_SatSwatch = fetchRGBParam("satSwatch");
	m_ShadSwatch = fetchRGBParam("shadSwatch");
	m_MidSwatch = fetchRGBParam("midSwatch");
	m_HighSwatch = fetchRGBParam("highSwatch");
	
	m_Exp = fetchDoubleParam("exp");
	m_ExpR = fetchDoubleParam("expR");
	m_ExpG = fetchDoubleParam("expG");
	m_ExpB = fetchDoubleParam("expB");
	m_Cont = fetchDoubleParam("cont");
	m_ContR = fetchDoubleParam("contR");
	m_ContG = fetchDoubleParam("contG");
	m_ContB = fetchDoubleParam("contB");
	m_Sat = fetchDoubleParam("sat");
	m_SatR = fetchDoubleParam("satR");
	m_SatG = fetchDoubleParam("satG");
	m_SatB = fetchDoubleParam("satB");
	m_Shad = fetchDoubleParam("shad");
	m_ShadR = fetchDoubleParam("shadR");
	m_ShadG = fetchDoubleParam("shadG");
	m_ShadB = fetchDoubleParam("shadB");
	m_Mid = fetchDoubleParam("mid");
	m_MidR = fetchDoubleParam("midR");
	m_MidG = fetchDoubleParam("midG");
	m_MidB = fetchDoubleParam("midB");
	m_High = fetchDoubleParam("high");
	m_HighR = fetchDoubleParam("highR");
	m_HighG = fetchDoubleParam("highG");
	m_HighB = fetchDoubleParam("highB");
	m_ShadP = fetchDoubleParam("shadP");
	m_ShadPP = fetchDoubleParam("shadPP");
	m_HighP = fetchDoubleParam("highP");
	m_HighPP = fetchDoubleParam("highPP");
	m_ContP = fetchDoubleParam("contP");
	m_ContPP = fetchDoubleParam("contPP");
	m_Display = fetchBooleanParam("display");
	
    
}

void FilmGradePlugin::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        ImageScaler imageScaler(*this);
        setupAndProcess(imageScaler, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool FilmGradePlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
    float exp = m_Exp->getValueAtTime(p_Args.time);
    float expR = m_ExpR->getValueAtTime(p_Args.time);
    float expG = m_ExpG->getValueAtTime(p_Args.time);
    float expB = m_ExpB->getValueAtTime(p_Args.time);
    float cont = m_Cont->getValueAtTime(p_Args.time);
    float contR = m_ContR->getValueAtTime(p_Args.time);
    float contG = m_ContG->getValueAtTime(p_Args.time);
    float contB = m_ContB->getValueAtTime(p_Args.time);
    float sat = m_Sat->getValueAtTime(p_Args.time);
    float satR = m_SatR->getValueAtTime(p_Args.time);
    float satG = m_SatG->getValueAtTime(p_Args.time);
    float satB = m_SatB->getValueAtTime(p_Args.time);
    float shad = m_Shad->getValueAtTime(p_Args.time);
    float shadR = m_ShadR->getValueAtTime(p_Args.time);
    float shadG = m_ShadG->getValueAtTime(p_Args.time);
    float shadB = m_ShadB->getValueAtTime(p_Args.time);
    float mid = m_Mid->getValueAtTime(p_Args.time);
    float midR = m_MidR->getValueAtTime(p_Args.time);
    float midG = m_MidG->getValueAtTime(p_Args.time);
    float midB = m_MidB->getValueAtTime(p_Args.time);
    float high = m_High->getValueAtTime(p_Args.time);
    float highR = m_HighR->getValueAtTime(p_Args.time);
    float highG = m_HighG->getValueAtTime(p_Args.time);
    float highB = m_HighB->getValueAtTime(p_Args.time);
    bool aDisplay = m_Display->getValueAtTime(p_Args.time);
    

    if ((exp == 0.0f) && (expR == 0.0f) && (expG == 0.0f) && (expB == 0.0f) && (cont == 1.0f) && (contR == 1.0f) && (contG == 1.0f) && (contB == 1.0f) && 
    (sat == 1.0f) && (satR == 1.0f) && (satG == 1.0f) && (satB == 1.0f) && (shad == 0.0f) && (shadR == 0.0f) && (shadG == 0.0f) && (shadB == 0.0f) && 
    (mid == 0.0f) && (midR == 0.0f) && (midG == 0.0f) && (midB == 0.0f) && (high == 0.0f) && (highR == 0.0f) && (highG == 0.0f) && (highB == 0.0f) && !(aDisplay))
    {
        p_IdentityClip = m_SrcClip;
        p_IdentityTime = p_Args.time;
        return true;
    }

    return false;
}

void FilmGradePlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
   		 	
  	if (p_ParamName == "exp" && p_Args.reason == OFX::eChangeUserEdit)
    {
    	
   		float exp = m_Exp->getValueAtTime(p_Args.time);
   		float expR = m_ExpR->getValueAtTime(p_Args.time);
    	float expG = m_ExpG->getValueAtTime(p_Args.time);
    	float expB = m_ExpB->getValueAtTime(p_Args.time);
    	
    	float exp1 = (expR + expG + expB)/3.0f;
    	float expr = expR + (exp - exp1);
    	float expg = expG + (exp - exp1);
    	float expb = expB + (exp - exp1);
    	
    	float expR1 = expr > 20.0f ? expr - 20.0f : expr < -20.0f ? expr + 20.0f : 0.0f;
    	float expG1 = expg > 20.0f ? expg - 20.0f : expg < -20.0f ? expg + 20.0f : 0.0f;
    	float expB1 = expb > 20.0f ? expb - 20.0f : expb < -20.0f ? expb + 20.0f : 0.0f;
    	
    	float expR2 = expR + (exp - exp1 - expR1 + expG1/2.0f + expB1/2.0f);
    	float expG2 = expG + (exp - exp1 - expG1 + expR1/2.0f + expB1/2.0f);
    	float expB2 = expB + (exp - exp1 - expB1 + expR1/2.0f + expG1/2.0f);
    	   	
    	float ExpSwatchR = expR2 >= expG2 && expR2 >= expB2 ? 1.0f : 1.0f - (fmax(expG2, expB2) - expR2)/40.0f;
    	float ExpSwatchG = expG2 >= expR2 && expG2 >= expB2 ? 1.0f : 1.0f - (fmax(expR2, expB2) - expG2)/40.0f;
    	float ExpSwatchB = expB2 >= expR2 && expB2 >= expG2 ? 1.0f : 1.0f - (fmax(expR2, expG2) - expB2)/40.0f;
    	
    	beginEditBlock("expR");
    	beginEditBlock("expG");
    	beginEditBlock("expB");
    	beginEditBlock("expSwatch");
    	
    	m_ExpR->setValue(expR2);
    	m_ExpG->setValue(expG2);
    	m_ExpB->setValue(expB2);
    	m_ExpSwatch->setValue(ExpSwatchR, ExpSwatchG, ExpSwatchB);
    	
    	endEditBlock();
    }
  	
    if (p_ParamName == "expR" && p_Args.reason == OFX::eChangeUserEdit)
    {

   		float expR = m_ExpR->getValueAtTime(p_Args.time);
    	float expG = m_ExpG->getValueAtTime(p_Args.time);
    	float expB = m_ExpB->getValueAtTime(p_Args.time);
    	
    	float Exp = (expR + expG + expB)/3.0f;
    	
    	float ExpSwatchR = expR >= expG && expR >= expB ? 1.0f : 1.0f - (fmax(expG, expB) - expR)/40.0f;
    	float ExpSwatchG = expG >= expR && expG >= expB ? 1.0f : 1.0f - (fmax(expR, expB) - expG)/40.0f;
    	float ExpSwatchB = expB >= expR && expB >= expG ? 1.0f : 1.0f - (fmax(expR, expG) - expB)/40.0f;
    	
    	beginEditBlock("exp");
    	beginEditBlock("expSwatch");
    	
    	m_Exp->setValue(Exp);
    	m_ExpSwatch->setValue(ExpSwatchR, ExpSwatchG, ExpSwatchB);
    	
    	endEditBlock();
    	
    }
    
     if (p_ParamName == "expG" && p_Args.reason == OFX::eChangeUserEdit)
    {
    	
   		float expR = m_ExpR->getValueAtTime(p_Args.time);
    	float expG = m_ExpG->getValueAtTime(p_Args.time);
    	float expB = m_ExpB->getValueAtTime(p_Args.time);
    	
    	float Exp = (expR + expG + expB)/3.0f;
    	
    	float ExpSwatchR = expR >= expG && expR >= expB ? 1.0f : 1.0f - (fmax(expG, expB) - expR)/40.0f;
    	float ExpSwatchG = expG >= expR && expG >= expB ? 1.0f : 1.0f - (fmax(expR, expB) - expG)/40.0f;
    	float ExpSwatchB = expB >= expR && expB >= expG ? 1.0f : 1.0f - (fmax(expR, expG) - expB)/40.0f;
    	
    	beginEditBlock("exp");
    	beginEditBlock("expSwatch");
    	
    	m_Exp->setValue(Exp);
    	m_ExpSwatch->setValue(ExpSwatchR, ExpSwatchG, ExpSwatchB);
    	
    	endEditBlock();
    	
    }
    
     if (p_ParamName == "expB" && p_Args.reason == OFX::eChangeUserEdit)
    {
    	
    	
   		float expR = m_ExpR->getValueAtTime(p_Args.time);
    	float expG = m_ExpG->getValueAtTime(p_Args.time);
    	float expB = m_ExpB->getValueAtTime(p_Args.time);
    	
    	
    	float Exp = (expR + expG + expB)/3.0f;
    	
    	float ExpSwatchR = expR >= expG && expR >= expB ? 1.0f : 1.0f - (fmax(expG, expB) - expR)/40.0f;
    	float ExpSwatchG = expG >= expR && expG >= expB ? 1.0f : 1.0f - (fmax(expR, expB) - expG)/40.0f;
    	float ExpSwatchB = expB >= expR && expB >= expG ? 1.0f : 1.0f - (fmax(expR, expG) - expB)/40.0f;
    	
    	beginEditBlock("exp");
    	beginEditBlock("expSwatch");
    	
    	m_Exp->setValue(Exp);
    	m_ExpSwatch->setValue(ExpSwatchR, ExpSwatchG, ExpSwatchB);
    	
    	endEditBlock();
    	
    }
    	
    
    if (p_ParamName == "expSwatch" && p_Args.reason == OFX::eChangeUserEdit)
    {
        RGBValues expSwatch;
   		m_ExpSwatch->getValueAtTime(p_Args.time, expSwatch.r, expSwatch.g, expSwatch.b);
   		float exp = m_Exp->getValueAtTime(p_Args.time);
        
    	float expr = exp + (expSwatch.r - (expSwatch.g + expSwatch.b)/2.0f) * (20.0f - sqrt(exp*exp));
    	float expg = exp + (expSwatch.g - (expSwatch.r + expSwatch.b)/2.0f) * (20.0f - sqrt(exp*exp));
    	float expb = exp + (expSwatch.b - (expSwatch.r + expSwatch.g)/2.0f) * (20.0f - sqrt(exp*exp));
    	
    	beginEditBlock("expR");
    	beginEditBlock("expG");
    	beginEditBlock("expB");
    	
    	m_ExpR->setValue(expr);
    	m_ExpG->setValue(expg);
    	m_ExpB->setValue(expb);
    	
    	endEditBlock();
    	
    }
    
     if (p_ParamName == "cont" && p_Args.reason == OFX::eChangeUserEdit)
    {
    
   		float cont = m_Cont->getValueAtTime(p_Args.time);
   		float contR = m_ContR->getValueAtTime(p_Args.time);
    	float contG = m_ContG->getValueAtTime(p_Args.time);
    	float contB = m_ContB->getValueAtTime(p_Args.time);
    	
    	float cont1 = (contR + contG + contB)/3.0f;
    	float contr = contR + (cont - cont1);
    	float contg = contG + (cont - cont1);
    	float contb = contB + (cont - cont1);
    	
    	float contR1 = contr > 3.0f ? contr - 3.0f : contr < 0.0f ? contr : 0.0f;
    	float contG1 = contg > 3.0f ? contg - 3.0f : contg < 0.0f ? contg : 0.0f;
    	float contB1 = contb > 3.0f ? contb - 3.0f : contb < 0.0f ? contb : 0.0f;
    	
    	float contR2 = contR + (cont - cont1 - contR1 + contG1/2.0f + contB1/2.0f);
    	float contG2 = contG + (cont - cont1 - contG1 + contR1/2.0f + contB1/2.0f);
    	float contB2 = contB + (cont - cont1 - contB1 + contR1/2.0f + contG1/2.0f);
    	
    	float ContSwatchR = contR2 >= contG2 && contR2 >= contB2 ? 1.0f : 1.0f - (fmax(contG2, contB2) - contR2)/3.0f;
    	float ContSwatchG = contG2 >= contR2 && contG2 >= contB2 ? 1.0f : 1.0f - (fmax(contR2, contB2) - contG2)/3.0f;
    	float ContSwatchB = contB2 >= contR2 && contB2 >= contG2 ? 1.0f : 1.0f - (fmax(contR2, contG2) - contB2)/3.0f;
    	
    	beginEditBlock("contR");
    	beginEditBlock("contG");
    	beginEditBlock("contB");
    	beginEditBlock("contSwatch");
    	
    	m_ContR->setValue(contR2);
    	m_ContG->setValue(contG2);
    	m_ContB->setValue(contB2);
    	m_ContSwatch->setValue(ContSwatchR, ContSwatchG, ContSwatchB);
    	
    	endEditBlock();
    	
    }
    
    if (p_ParamName == "contR" && p_Args.reason == OFX::eChangeUserEdit)
    {
    	
   		float contR = m_ContR->getValueAtTime(p_Args.time);
    	float contG = m_ContG->getValueAtTime(p_Args.time);
    	float contB = m_ContB->getValueAtTime(p_Args.time);
    	
    	float Cont = (contR + contG + contB)/3.0f;
    	
    	float ContSwatchR = contR >= contG && contR >= contB ? 1.0f : 1.0f - (fmax(contG, contB) - contR)/3.0f;
    	float ContSwatchG = contG >= contR && contG >= contB ? 1.0f : 1.0f - (fmax(contR, contB) - contG)/3.0f;
    	float ContSwatchB = contB >= contR && contB >= contG ? 1.0f : 1.0f - (fmax(contR, contG) - contB)/3.0f;
    	
    	beginEditBlock("cont");
    	beginEditBlock("contSwatch");
    	
    	m_Cont->setValue(Cont);
    	m_ContSwatch->setValue(ContSwatchR, ContSwatchG, ContSwatchB);
    	
    	endEditBlock();
    	
    }
    
     if (p_ParamName == "contG" && p_Args.reason == OFX::eChangeUserEdit)
    {
    
   		float contR = m_ContR->getValueAtTime(p_Args.time);
    	float contG = m_ContG->getValueAtTime(p_Args.time);
    	float contB = m_ContB->getValueAtTime(p_Args.time);
    	
    	float Cont = (contR + contG + contB)/3.0f;
    	
    	float ContSwatchR = contR >= contG && contR >= contB ? 1.0f : 1.0f - (fmax(contG, contB) - contR)/3.0f;
    	float ContSwatchG = contG >= contR && contG >= contB ? 1.0f : 1.0f - (fmax(contR, contB) - contG)/3.0f;
    	float ContSwatchB = contB >= contR && contB >= contG ? 1.0f : 1.0f - (fmax(contR, contG) - contB)/3.0f;
    	
    	beginEditBlock("cont");
    	beginEditBlock("contSwatch");
    	
    	m_Cont->setValue(Cont);
    	m_ContSwatch->setValue(ContSwatchR, ContSwatchG, ContSwatchB);
    	
    	endEditBlock();
    	
    }
    
     if (p_ParamName == "contB" && p_Args.reason == OFX::eChangeUserEdit)
    {
    	
   		float contR = m_ContR->getValueAtTime(p_Args.time);
    	float contG = m_ContG->getValueAtTime(p_Args.time);
    	float contB = m_ContB->getValueAtTime(p_Args.time);
    	
    	float Cont = (contR + contG + contB)/3.0f;
    	
    	float ContSwatchR = contR >= contG && contR >= contB ? 1.0f : 1.0f - (fmax(contG, contB) - contR)/3.0f;
    	float ContSwatchG = contG >= contR && contG >= contB ? 1.0f : 1.0f - (fmax(contR, contB) - contG)/3.0f;
    	float ContSwatchB = contB >= contR && contB >= contG ? 1.0f : 1.0f - (fmax(contR, contG) - contB)/3.0f;
    	
    	beginEditBlock("cont");
    	beginEditBlock("contSwatch");
    	
    	m_Cont->setValue(Cont);
    	m_ContSwatch->setValue(ContSwatchR, ContSwatchG, ContSwatchB);
    	
    	endEditBlock();
    	
    }
    
    if (p_ParamName == "contSwatch" && p_Args.reason == OFX::eChangeUserEdit)
    {
        RGBValues contSwatch;
   		m_ContSwatch->getValueAtTime(p_Args.time, contSwatch.r, contSwatch.g, contSwatch.b);
   		float cont = m_Cont->getValueAtTime(p_Args.time);
        
        float cont1 = cont >= 1.0f ? (3.0f - cont) : cont;
    	float contr = cont + (contSwatch.r - (contSwatch.g + contSwatch.b)/2.0f) * cont1;
    	float contg = cont + (contSwatch.g - (contSwatch.r + contSwatch.b)/2.0f) * cont1;
    	float contb = cont + (contSwatch.b - (contSwatch.r + contSwatch.g)/2.0f) * cont1;
    	
    	beginEditBlock("contR");
    	beginEditBlock("contG");
    	beginEditBlock("contB");
    	
    	m_ContR->setValue(contr);
    	m_ContG->setValue(contg);
    	m_ContB->setValue(contb);
    	
    	endEditBlock();
    	
    }
    
    
     if (p_ParamName == "sat" && p_Args.reason == OFX::eChangeUserEdit)
    {
    
   		float sat = m_Sat->getValueAtTime(p_Args.time);
   		float satR = m_SatR->getValueAtTime(p_Args.time);
    	float satG = m_SatG->getValueAtTime(p_Args.time);
    	float satB = m_SatB->getValueAtTime(p_Args.time);
    	
    	float sat1 = (satR + satG + satB)/3.0f;
    	float satr = satR + (sat - sat1);
    	float satg = satG + (sat - sat1);
    	float satb = satB + (sat - sat1);
    	
    	float satR1 = satr > 3.0f ? satr - 3.0f : satr < 0.0f ? satr : 0.0f;
    	float satG1 = satg > 3.0f ? satg - 3.0f : satg < 0.0f ? satg : 0.0f;
    	float satB1 = satb > 3.0f ? satb - 3.0f : satb < 0.0f ? satb : 0.0f;
    	
    	float satR2 = satR + (sat - sat1 - satR1 + satG1/2.0f + satB1/2.0f);
    	float satG2 = satG + (sat - sat1 - satG1 + satR1/2.0f + satB1/2.0f);
    	float satB2 = satB + (sat - sat1 - satB1 + satR1/2.0f + satG1/2.0f);
    	
    	float SatSwatchR = satR2 >= satG2 && satR2 >= satB2 ? 1.0f : 1.0f - (fmax(satG2, satB2) - satR2)/3.0f;
    	float SatSwatchG = satG2 >= satR2 && satG2 >= satB2 ? 1.0f : 1.0f - (fmax(satR2, satB2) - satG2)/3.0f;
    	float SatSwatchB = satB2 >= satR2 && satB2 >= satG2 ? 1.0f : 1.0f - (fmax(satR2, satG2) - satB2)/3.0f;
    
    	beginEditBlock("satR");
    	beginEditBlock("satG");
    	beginEditBlock("satB");
    	beginEditBlock("satSwatch");
    	
    	m_SatR->setValue(satR2);
    	m_SatG->setValue(satG2);
    	m_SatB->setValue(satB2);
    	m_SatSwatch->setValue(SatSwatchR, SatSwatchG, SatSwatchB);
    	
    	endEditBlock();
    	
    }
    
    if (p_ParamName == "satR" && p_Args.reason == OFX::eChangeUserEdit)
    {
    	
   		float satR = m_SatR->getValueAtTime(p_Args.time);
    	float satG = m_SatG->getValueAtTime(p_Args.time);
    	float satB = m_SatB->getValueAtTime(p_Args.time);
    	
    	float Sat = (satR + satG + satB)/3.0f;
    	
    	float SatSwatchR = satR >= satG && satR >= satB ? 1.0f : 1.0f - (fmax(satG, satB) - satR)/3.0f;
    	float SatSwatchG = satG >= satR && satG >= satB ? 1.0f : 1.0f - (fmax(satR, satB) - satG)/3.0f;
    	float SatSwatchB = satB >= satR && satB >= satG ? 1.0f : 1.0f - (fmax(satR, satG) - satB)/3.0f;
    	
    	beginEditBlock("sat");
    	beginEditBlock("satSwatch");
    	
    	m_Sat->setValue(Sat);
    	m_SatSwatch->setValue(SatSwatchR, SatSwatchG, SatSwatchB);
    	
    	endEditBlock();
    	
    }
    
     if (p_ParamName == "satG" && p_Args.reason == OFX::eChangeUserEdit)
    {
    
   		float satR = m_SatR->getValueAtTime(p_Args.time);
    	float satG = m_SatG->getValueAtTime(p_Args.time);
    	float satB = m_SatB->getValueAtTime(p_Args.time);
    	
    	float Sat = (satR + satG + satB)/3.0f;
    	
    	float SatSwatchR = satR >= satG && satR >= satB ? 1.0f : 1.0f - (fmax(satG, satB) - satR)/3.0f;
    	float SatSwatchG = satG >= satR && satG >= satB ? 1.0f : 1.0f - (fmax(satR, satB) - satG)/3.0f;
    	float SatSwatchB = satB >= satR && satB >= satG ? 1.0f : 1.0f - (fmax(satR, satG) - satB)/3.0f;
    	
    	beginEditBlock("sat");
    	beginEditBlock("satSwatch");
    	
    	m_Sat->setValue(Sat);
    	m_SatSwatch->setValue(SatSwatchR, SatSwatchG, SatSwatchB);
    	
    	endEditBlock();
    	
    }
    
     if (p_ParamName == "satB" && p_Args.reason == OFX::eChangeUserEdit)
    {
    	
   		float satR = m_SatR->getValueAtTime(p_Args.time);
    	float satG = m_SatG->getValueAtTime(p_Args.time);
    	float satB = m_SatB->getValueAtTime(p_Args.time);
    	
    	float Sat = (satR + satG + satB)/3.0f;
    	
    	float SatSwatchR = satR >= satG && satR >= satB ? 1.0f : 1.0f - (fmax(satG, satB) - satR)/3.0f;
    	float SatSwatchG = satG >= satR && satG >= satB ? 1.0f : 1.0f - (fmax(satR, satB) - satG)/3.0f;
    	float SatSwatchB = satB >= satR && satB >= satG ? 1.0f : 1.0f - (fmax(satR, satG) - satB)/3.0f;
    	
    	beginEditBlock("sat");
    	beginEditBlock("satSwatch");
    	
    	m_Sat->setValue(Sat);
    	m_SatSwatch->setValue(SatSwatchR, SatSwatchG, SatSwatchB);
    	
    	endEditBlock();
    	
    }
    
    if (p_ParamName == "satSwatch" && p_Args.reason == OFX::eChangeUserEdit)
    {
        RGBValues satSwatch;
   		m_SatSwatch->getValueAtTime(p_Args.time, satSwatch.r, satSwatch.g, satSwatch.b);
   		float sat = m_Sat->getValueAtTime(p_Args.time);
        
        float sat1 = sat >= 1.0f ? (3.0f - sat) : sat;
    	float satr = sat + (satSwatch.r - (satSwatch.g + satSwatch.b)/2.0f) * sat1;
    	float satg = sat + (satSwatch.g - (satSwatch.r + satSwatch.b)/2.0f) * sat1;
    	float satb = sat + (satSwatch.b - (satSwatch.r + satSwatch.g)/2.0f) * sat1;
    	
    	beginEditBlock("satR");
    	beginEditBlock("satG");
    	beginEditBlock("satB");
    	
    	m_SatR->setValue(satr);
    	m_SatG->setValue(satg);
    	m_SatB->setValue(satb);
    	
    	endEditBlock();
    	
    }
    
    
     if (p_ParamName == "mid" && p_Args.reason == OFX::eChangeUserEdit)
    {
    
   		float mid = m_Mid->getValueAtTime(p_Args.time);
   		float midR = m_MidR->getValueAtTime(p_Args.time);
    	float midG = m_MidG->getValueAtTime(p_Args.time);
    	float midB = m_MidB->getValueAtTime(p_Args.time);
    	
    	float mid1 = (midR + midG + midB)/3.0f;
    	float midr = midR + (mid - mid1);
    	float midg = midG + (mid - mid1);
    	float midb = midB + (mid - mid1);
    	
    	float midR1 = midr > 0.5f ? midr - 0.5f : midr < -0.5f ? midr + 0.5f : 0.0f;
    	float midG1 = midg > 0.5f ? midg - 0.5f : midg < -0.5f ? midg + 0.5f : 0.0f;
    	float midB1 = midb > 0.5f ? midb - 0.5f : midb < -0.5f ? midb + 0.5f : 0.0f;
    	
    	float midR2 = midR + (mid - mid1 - midR1 + midG1/2.0f + midB1/2.0f);
    	float midG2 = midG + (mid - mid1 - midG1 + midR1/2.0f + midB1/2.0f);
    	float midB2 = midB + (mid - mid1 - midB1 + midR1/2.0f + midG1/2.0f);
    	
    	float MidSwatchR = midR2 >= midG2 && midR2 >= midB2 ? 1.0f : 1.0f - (fmax(midG2, midB2) - midR2);
    	float MidSwatchG = midG2 >= midR2 && midG2 >= midB2 ? 1.0f : 1.0f - (fmax(midR2, midB2) - midG2);
    	float MidSwatchB = midB2 >= midR2 && midB2 >= midG2 ? 1.0f : 1.0f - (fmax(midR2, midG2) - midB2);
    	
    	beginEditBlock("midR");
    	beginEditBlock("midG");
    	beginEditBlock("midB");
    	beginEditBlock("midSwatch");
    	
    	m_MidR->setValue(midR2);
    	m_MidG->setValue(midG2);
    	m_MidB->setValue(midB2);
    	m_MidSwatch->setValue(MidSwatchR, MidSwatchG, MidSwatchB);
    	
    	endEditBlock();
    	
    }
    
    if (p_ParamName == "midR" && p_Args.reason == OFX::eChangeUserEdit)
    {
    	
   		float midR = m_MidR->getValueAtTime(p_Args.time);
    	float midG = m_MidG->getValueAtTime(p_Args.time);
    	float midB = m_MidB->getValueAtTime(p_Args.time);
    	
    	float Mid = (midR + midG + midB)/3.0f;
    	
    	float MidSwatchR = midR >= midG && midR >= midB ? 1.0f : 1.0f - (fmax(midG, midB) - midR);
    	float MidSwatchG = midG >= midR && midG >= midB ? 1.0f : 1.0f - (fmax(midR, midB) - midG);
    	float MidSwatchB = midB >= midR && midB >= midG ? 1.0f : 1.0f - (fmax(midR, midG) - midB);
    	
    	beginEditBlock("mid");
    	beginEditBlock("midSwatch");
    	
    	m_Mid->setValue(Mid);
    	m_MidSwatch->setValue(MidSwatchR, MidSwatchG, MidSwatchB);
    	
    	endEditBlock();
    	
    }
    
     if (p_ParamName == "midG" && p_Args.reason == OFX::eChangeUserEdit)
    {
    
   		float midR = m_MidR->getValueAtTime(p_Args.time);
    	float midG = m_MidG->getValueAtTime(p_Args.time);
    	float midB = m_MidB->getValueAtTime(p_Args.time);
    	
    	float Mid = (midR + midG + midB)/3.0f;
    	
    	float MidSwatchR = midR >= midG && midR >= midB ? 1.0f : 1.0f - (fmax(midG, midB) - midR);
    	float MidSwatchG = midG >= midR && midG >= midB ? 1.0f : 1.0f - (fmax(midR, midB) - midG);
    	float MidSwatchB = midB >= midR && midB >= midG ? 1.0f : 1.0f - (fmax(midR, midG) - midB);
    	
    	beginEditBlock("mid");
    	beginEditBlock("midSwatch");
    	
    	m_Mid->setValue(Mid);
    	m_MidSwatch->setValue(MidSwatchR, MidSwatchG, MidSwatchB);
    	
    	endEditBlock();
    	
    }
    
     if (p_ParamName == "midB" && p_Args.reason == OFX::eChangeUserEdit)
    {
    	
   		float midR = m_MidR->getValueAtTime(p_Args.time);
    	float midG = m_MidG->getValueAtTime(p_Args.time);
    	float midB = m_MidB->getValueAtTime(p_Args.time);
    	
    	float Mid = (midR + midG + midB)/3.0f;
    	
    	float MidSwatchR = midR >= midG && midR >= midB ? 1.0f : 1.0f - (fmax(midG, midB) - midR);
    	float MidSwatchG = midG >= midR && midG >= midB ? 1.0f : 1.0f - (fmax(midR, midB) - midG);
    	float MidSwatchB = midB >= midR && midB >= midG ? 1.0f : 1.0f - (fmax(midR, midG) - midB);
    	
    	beginEditBlock("mid");
    	beginEditBlock("midSwatch");
    	
    	m_Mid->setValue(Mid);
    	m_MidSwatch->setValue(MidSwatchR, MidSwatchG, MidSwatchB);
    	
    	endEditBlock();
    	
    }
    
    if (p_ParamName == "midSwatch" && p_Args.reason == OFX::eChangeUserEdit)
    {
        RGBValues midSwatch;
   		m_MidSwatch->getValueAtTime(p_Args.time, midSwatch.r, midSwatch.g, midSwatch.b);
   		float mid = m_Mid->getValueAtTime(p_Args.time);
        
        float midr = mid + (midSwatch.r - (midSwatch.g + midSwatch.b)/2.0f) * (0.5f - sqrt(mid*mid));
    	float midg = mid + (midSwatch.g - (midSwatch.r + midSwatch.b)/2.0f) * (0.5f - sqrt(mid*mid));
    	float midb = mid + (midSwatch.b - (midSwatch.r + midSwatch.g)/2.0f) * (0.5f - sqrt(mid*mid));
    	
    	beginEditBlock("midR");
    	beginEditBlock("midG");
    	beginEditBlock("midB");
    	
    	m_MidR->setValue(midr);
    	m_MidG->setValue(midg);
    	m_MidB->setValue(midb);
    	
    	endEditBlock();
    	
    }
   
   
     if (p_ParamName == "shad" && p_Args.reason == OFX::eChangeUserEdit)
    {
    
   		float shad = m_Shad->getValueAtTime(p_Args.time);
   		float shadR = m_ShadR->getValueAtTime(p_Args.time);
    	float shadG = m_ShadG->getValueAtTime(p_Args.time);
    	float shadB = m_ShadB->getValueAtTime(p_Args.time);
    	
    	float shad1 = (shadR + shadG + shadB)/3.0f;
    	float shadr = shadR + (shad - shad1);
    	float shadg = shadG + (shad - shad1);
    	float shadb = shadB + (shad - shad1);
    	
    	float shadR1 = shadr > 0.5f ? shadr - 0.5f : shadr < -0.5f ? shadr + 0.5f : 0.0f;
    	float shadG1 = shadg > 0.5f ? shadg - 0.5f : shadg < -0.5f ? shadg + 0.5f : 0.0f;
    	float shadB1 = shadb > 0.5f ? shadb - 0.5f : shadb < -0.5f ? shadb + 0.5f : 0.0f;
    	
    	float shadR2 = shadR + (shad - shad1 - shadR1 + shadG1/2.0f + shadB1/2.0f);
    	float shadG2 = shadG + (shad - shad1 - shadG1 + shadR1/2.0f + shadB1/2.0f);
    	float shadB2 = shadB + (shad - shad1 - shadB1 + shadR1/2.0f + shadG1/2.0f);
    	
    	float ShadSwatchR = shadR2 >= shadG2 && shadR2 >= shadB2 ? 1.0f : 1.0f - (fmax(shadG2, shadB2) - shadR2);
    	float ShadSwatchG = shadG2 >= shadR2 && shadG2 >= shadB2 ? 1.0f : 1.0f - (fmax(shadR2, shadB2) - shadG2);
    	float ShadSwatchB = shadB2 >= shadR2 && shadB2 >= shadG2 ? 1.0f : 1.0f - (fmax(shadR2, shadG2) - shadB2);
    	
    	beginEditBlock("shadR");
    	beginEditBlock("shadG");
    	beginEditBlock("shadB");
    	beginEditBlock("shadSwatch");
    	
    	m_ShadR->setValue(shadR2);
    	m_ShadG->setValue(shadG2);
    	m_ShadB->setValue(shadB2);
    	m_ShadSwatch->setValue(ShadSwatchR, ShadSwatchG, ShadSwatchB);
    	
    	endEditBlock();
    	
    }
    
    if (p_ParamName == "shadR" && p_Args.reason == OFX::eChangeUserEdit)
    {
    	
   		float shadR = m_ShadR->getValueAtTime(p_Args.time);
    	float shadG = m_ShadG->getValueAtTime(p_Args.time);
    	float shadB = m_ShadB->getValueAtTime(p_Args.time);
    	
    	float Shad = (shadR + shadG + shadB)/3.0f;
    	
    	float ShadSwatchR = shadR >= shadG && shadR >= shadB ? 1.0f : 1.0f - (fmax(shadG, shadB) - shadR);
    	float ShadSwatchG = shadG >= shadR && shadG >= shadB ? 1.0f : 1.0f - (fmax(shadR, shadB) - shadG);
    	float ShadSwatchB = shadB >= shadR && shadB >= shadG ? 1.0f : 1.0f - (fmax(shadR, shadG) - shadB);
    	
    	beginEditBlock("shad");
    	beginEditBlock("shadSwatch");
    	
    	m_Shad->setValue(Shad);
    	m_ShadSwatch->setValue(ShadSwatchR, ShadSwatchG, ShadSwatchB);
    	
    	endEditBlock();
    	
    }
    
     if (p_ParamName == "shadG" && p_Args.reason == OFX::eChangeUserEdit)
    {
    
   		float shadR = m_ShadR->getValueAtTime(p_Args.time);
    	float shadG = m_ShadG->getValueAtTime(p_Args.time);
    	float shadB = m_ShadB->getValueAtTime(p_Args.time);
    	
    	float Shad = (shadR + shadG + shadB)/3.0f;
    	
    	float ShadSwatchR = shadR >= shadG && shadR >= shadB ? 1.0f : 1.0f - (fmax(shadG, shadB) - shadR);
    	float ShadSwatchG = shadG >= shadR && shadG >= shadB ? 1.0f : 1.0f - (fmax(shadR, shadB) - shadG);
    	float ShadSwatchB = shadB >= shadR && shadB >= shadG ? 1.0f : 1.0f - (fmax(shadR, shadG) - shadB);
    	
    	beginEditBlock("shad");
    	beginEditBlock("shadSwatch");
    	
    	m_Shad->setValue(Shad);
    	m_ShadSwatch->setValue(ShadSwatchR, ShadSwatchG, ShadSwatchB);
    	
    	endEditBlock();
    	
    }
    
     if (p_ParamName == "shadB" && p_Args.reason == OFX::eChangeUserEdit)
    {
    	
   		float shadR = m_ShadR->getValueAtTime(p_Args.time);
    	float shadG = m_ShadG->getValueAtTime(p_Args.time);
    	float shadB = m_ShadB->getValueAtTime(p_Args.time);
    	
    	float Shad = (shadR + shadG + shadB)/3.0f;
    	
    	float ShadSwatchR = shadR >= shadG && shadR >= shadB ? 1.0f : 1.0f - (fmax(shadG, shadB) - shadR);
    	float ShadSwatchG = shadG >= shadR && shadG >= shadB ? 1.0f : 1.0f - (fmax(shadR, shadB) - shadG);
    	float ShadSwatchB = shadB >= shadR && shadB >= shadG ? 1.0f : 1.0f - (fmax(shadR, shadG) - shadB);
    	
    	beginEditBlock("shad");
    	beginEditBlock("shadSwatch");
    	
    	m_Shad->setValue(Shad);
    	m_ShadSwatch->setValue(ShadSwatchR, ShadSwatchG, ShadSwatchB);
    	
    	endEditBlock();
    	
    }
    
    if (p_ParamName == "shadSwatch" && p_Args.reason == OFX::eChangeUserEdit)
    {
        RGBValues shadSwatch;
   		m_ShadSwatch->getValueAtTime(p_Args.time, shadSwatch.r, shadSwatch.g, shadSwatch.b);
   		float shad = m_Shad->getValueAtTime(p_Args.time);
        
        float shadr = shad + (shadSwatch.r - (shadSwatch.g + shadSwatch.b)/2.0f) * (0.5f - sqrt(shad*shad));
    	float shadg = shad + (shadSwatch.g - (shadSwatch.r + shadSwatch.b)/2.0f) * (0.5f - sqrt(shad*shad));
    	float shadb = shad + (shadSwatch.b - (shadSwatch.r + shadSwatch.g)/2.0f) * (0.5f - sqrt(shad*shad));
    	
    	beginEditBlock("shadR");
    	beginEditBlock("shadG");
    	beginEditBlock("shadB");
    	
    	m_ShadR->setValue(shadr);
    	m_ShadG->setValue(shadg);
    	m_ShadB->setValue(shadb);
    	
    	endEditBlock();
    	
    }
   
   
     if (p_ParamName == "high" && p_Args.reason == OFX::eChangeUserEdit)
    {
    
   		float high = m_High->getValueAtTime(p_Args.time);
   		float highR = m_HighR->getValueAtTime(p_Args.time);
    	float highG = m_HighG->getValueAtTime(p_Args.time);
    	float highB = m_HighB->getValueAtTime(p_Args.time);
    	
    	float high1 = (highR + highG + highB)/3.0f;
    	float highr = highR + (high - high1);
    	float highg = highG + (high - high1);
    	float highb = highB + (high - high1);
    	
    	float highR1 = highr > 0.5f ? highr - 0.5f : highr < -0.5f ? highr + 0.5f : 0.0f;
    	float highG1 = highg > 0.5f ? highg - 0.5f : highg < -0.5f ? highg + 0.5f : 0.0f;
    	float highB1 = highb > 0.5f ? highb - 0.5f : highb < -0.5f ? highb + 0.5f : 0.0f;
    	
    	float highR2 = highR + (high - high1 - highR1 + highG1/2.0f + highB1/2.0f);
    	float highG2 = highG + (high - high1 - highG1 + highR1/2.0f + highB1/2.0f);
    	float highB2 = highB + (high - high1 - highB1 + highR1/2.0f + highG1/2.0f);
    	
    	float HighSwatchR = highR2 >= highG2 && highR2 >= highB2 ? 1.0f : 1.0f - (fmax(highG2, highB2) - highR2);
    	float HighSwatchG = highG2 >= highR2 && highG2 >= highB2 ? 1.0f : 1.0f - (fmax(highR2, highB2) - highG2);
    	float HighSwatchB = highB2 >= highR2 && highB2 >= highG2 ? 1.0f : 1.0f - (fmax(highR2, highG2) - highB2);
    	
    	beginEditBlock("highR");
    	beginEditBlock("highG");
    	beginEditBlock("highB");
    	beginEditBlock("highSwatch");
    	
    	m_HighR->setValue(highR2);
    	m_HighG->setValue(highG2);
    	m_HighB->setValue(highB2);
    	m_HighSwatch->setValue(HighSwatchR, HighSwatchG, HighSwatchB);
    	
    	endEditBlock();
    	
    }
    
    if (p_ParamName == "highR" && p_Args.reason == OFX::eChangeUserEdit)
    {
    	
   		float highR = m_HighR->getValueAtTime(p_Args.time);
    	float highG = m_HighG->getValueAtTime(p_Args.time);
    	float highB = m_HighB->getValueAtTime(p_Args.time);
    	
    	float High = (highR + highG + highB)/3.0f;
    	
    	float HighSwatchR = highR >= highG && highR >= highB ? 1.0f : 1.0f - (fmax(highG, highB) - highR);
    	float HighSwatchG = highG >= highR && highG >= highB ? 1.0f : 1.0f - (fmax(highR, highB) - highG);
    	float HighSwatchB = highB >= highR && highB >= highG ? 1.0f : 1.0f - (fmax(highR, highG) - highB);
    	
    	beginEditBlock("high");
    	beginEditBlock("highSwatch");
    	
    	m_High->setValue(High);
    	m_HighSwatch->setValue(HighSwatchR, HighSwatchG, HighSwatchB);
    	
    	endEditBlock();
    	
    }
    
     if (p_ParamName == "highG" && p_Args.reason == OFX::eChangeUserEdit)
    {
    
   		float highR = m_HighR->getValueAtTime(p_Args.time);
    	float highG = m_HighG->getValueAtTime(p_Args.time);
    	float highB = m_HighB->getValueAtTime(p_Args.time);
    	
    	float High = (highR + highG + highB)/3.0f;
    	
    	float HighSwatchR = highR >= highG && highR >= highB ? 1.0f : 1.0f - (fmax(highG, highB) - highR);
    	float HighSwatchG = highG >= highR && highG >= highB ? 1.0f : 1.0f - (fmax(highR, highB) - highG);
    	float HighSwatchB = highB >= highR && highB >= highG ? 1.0f : 1.0f - (fmax(highR, highG) - highB);
    	
    	beginEditBlock("high");
    	beginEditBlock("highSwatch");
    	
    	m_High->setValue(High);
    	m_HighSwatch->setValue(HighSwatchR, HighSwatchG, HighSwatchB);
    	
    	endEditBlock();
    	
    }
    
     if (p_ParamName == "highB" && p_Args.reason == OFX::eChangeUserEdit)
    {
    	
   		float highR = m_HighR->getValueAtTime(p_Args.time);
    	float highG = m_HighG->getValueAtTime(p_Args.time);
    	float highB = m_HighB->getValueAtTime(p_Args.time);
    	
    	float High = (highR + highG + highB)/3.0f;
    	
    	float HighSwatchR = highR >= highG && highR >= highB ? 1.0f : 1.0f - (fmax(highG, highB) - highR);
    	float HighSwatchG = highG >= highR && highG >= highB ? 1.0f : 1.0f - (fmax(highR, highB) - highG);
    	float HighSwatchB = highB >= highR && highB >= highG ? 1.0f : 1.0f - (fmax(highR, highG) - highB);
    	
    	beginEditBlock("high");
    	beginEditBlock("highSwatch");
    	
    	m_High->setValue(High);
    	m_HighSwatch->setValue(HighSwatchR, HighSwatchG, HighSwatchB);
    	
    	endEditBlock();
    	
    }
    
    if (p_ParamName == "highSwatch" && p_Args.reason == OFX::eChangeUserEdit)
    {
        RGBValues highSwatch;
   		m_HighSwatch->getValueAtTime(p_Args.time, highSwatch.r, highSwatch.g, highSwatch.b);
   		float high = m_High->getValueAtTime(p_Args.time);
        
        float highr = high + (highSwatch.r - (highSwatch.g + highSwatch.b)/2.0f) * (0.5f - sqrt(high*high));
    	float highg = high + (highSwatch.g - (highSwatch.r + highSwatch.b)/2.0f) * (0.5f - sqrt(high*high));
    	float highb = high + (highSwatch.b - (highSwatch.r + highSwatch.g)/2.0f) * (0.5f - sqrt(high*high));
    	
    	beginEditBlock("highR");
    	beginEditBlock("highG");
    	beginEditBlock("highB");
    	
    	m_HighR->setValue(highr);
    	m_HighG->setValue(highg);
    	m_HighB->setValue(highb);
    	
    	endEditBlock();
    	
    }
    
    if (p_ParamName == "shadP" && p_Args.reason == OFX::eChangeUserEdit)
    {
    	float shadP = m_ShadP->getValueAtTime(p_Args.time);
    	m_ShadPP->setValue(shadP);
    	
    }
    
    if (p_ParamName == "shadPP" && p_Args.reason == OFX::eChangeUserEdit)
    {
    	float shadPP = m_ShadPP->getValueAtTime(p_Args.time);
    	m_ShadP->setValue(shadPP);
    	
    }
    
    if (p_ParamName == "highP" && p_Args.reason == OFX::eChangeUserEdit)
    {
    	float highP = m_HighP->getValueAtTime(p_Args.time);
    	m_HighPP->setValue(highP);
    	
    }
    
    if (p_ParamName == "highPP" && p_Args.reason == OFX::eChangeUserEdit)
    {
    	float highPP = m_HighPP->getValueAtTime(p_Args.time);
    	m_HighP->setValue(highPP);
    	
    }
    
    if (p_ParamName == "contP" && p_Args.reason == OFX::eChangeUserEdit)
    {
    	float contP = m_ContP->getValueAtTime(p_Args.time);
    	m_ContPP->setValue(contP);
    	
    }
    
    if (p_ParamName == "contPP" && p_Args.reason == OFX::eChangeUserEdit)
    {
    	float contPP = m_ContPP->getValueAtTime(p_Args.time);
    	m_ContP->setValue(contPP);
    	
    }
   
}
         
void FilmGradePlugin::setupAndProcess(ImageScaler& p_ImageScaler, const OFX::RenderArguments& p_Args)
{
    // Get the dst image
    std::auto_ptr<OFX::Image> dst(m_DstClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum dstBitDepth = dst->getPixelDepth();
    OFX::PixelComponentEnum dstComponents = dst->getPixelComponents();

    // Get the src image
    std::auto_ptr<OFX::Image> src(m_SrcClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum srcBitDepth = src->getPixelDepth();
    OFX::PixelComponentEnum srcComponents = src->getPixelComponents();

    // Check to see if the bit depth and number of components are the same
    if ((srcBitDepth != dstBitDepth) || (srcComponents != dstComponents))
    {
        OFX::throwSuiteStatusException(kOfxStatErrValue);
    }

    float expR = m_ExpR->getValueAtTime(p_Args.time);
    float expG = m_ExpG->getValueAtTime(p_Args.time);
    float expB = m_ExpB->getValueAtTime(p_Args.time);
    float contR = m_ContR->getValueAtTime(p_Args.time);
    float contG = m_ContG->getValueAtTime(p_Args.time);
    float contB = m_ContB->getValueAtTime(p_Args.time);
    float satR = m_SatR->getValueAtTime(p_Args.time);
    float satG = m_SatG->getValueAtTime(p_Args.time);
    float satB = m_SatB->getValueAtTime(p_Args.time);
    float shadR = m_ShadR->getValueAtTime(p_Args.time);
    float shadG = m_ShadG->getValueAtTime(p_Args.time);
    float shadB = m_ShadB->getValueAtTime(p_Args.time);
    float midR = m_MidR->getValueAtTime(p_Args.time);
    float midG = m_MidG->getValueAtTime(p_Args.time);
    float midB = m_MidB->getValueAtTime(p_Args.time);
    float highR = m_HighR->getValueAtTime(p_Args.time);
    float highG = m_HighG->getValueAtTime(p_Args.time);
    float highB = m_HighB->getValueAtTime(p_Args.time);
    float shadP = m_ShadP->getValueAtTime(p_Args.time)/1023;
    float highP = m_HighP->getValueAtTime(p_Args.time)/1023;
    float contP = m_ContP->getValueAtTime(p_Args.time)/1023;
    
    bool aDisplay = m_Display->getValueAtTime(p_Args.time);
    float display = (aDisplay) ? 1.0f : 0.0f; 
    
    // Set the images
    p_ImageScaler.setDstImg(dst.get());
    p_ImageScaler.setSrcImg(src.get());

    // Setup OpenCL and CUDA Render arguments
    p_ImageScaler.setGPURenderArgs(p_Args);

    // Set the render window
    p_ImageScaler.setRenderWindow(p_Args.renderWindow);

    // Set the scales
    p_ImageScaler.setScales(expR, expG, expB, contR, contG, contB, satR, satG, satB, 
    shadR, shadG, shadB, midR, midG, midB, highR, highG, highB, shadP, highP, contP, display);

    // Call the base class process member, this will call the derived templated process code
    p_ImageScaler.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

FilmGradePluginFactory::FilmGradePluginFactory()
    : OFX::PluginFactoryHelper<FilmGradePluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void FilmGradePluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
{
    // Basic labels
    p_Desc.setLabels(kPluginName, kPluginName, kPluginName);
    p_Desc.setPluginGrouping(kPluginGrouping);
    p_Desc.setPluginDescription(kPluginDescription);

    // Add the supported contexts, only filter at the moment
    p_Desc.addSupportedContext(eContextFilter);
    p_Desc.addSupportedContext(eContextGeneral);

    // Add supported pixel depths
    p_Desc.addSupportedBitDepth(eBitDepthFloat);

    // Set a few flags
    p_Desc.setSingleInstance(false);
    p_Desc.setHostFrameThreading(false);
    p_Desc.setSupportsMultiResolution(kSupportsMultiResolution);
    p_Desc.setSupportsTiles(kSupportsTiles);
    p_Desc.setTemporalClipAccess(false);
    p_Desc.setRenderTwiceAlways(false);
    p_Desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);

    // Setup OpenCL and CUDA render capability flags
    p_Desc.setSupportsOpenCLRender(true);
    p_Desc.setSupportsCudaRender(true);
}

static DoubleParamDescriptor* defineScaleParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, const std::string& p_Label,
                                               const std::string& p_Hint, GroupParamDescriptor* p_Parent)
{
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
	param->setLabel(p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);

    if (p_Parent)
    {
        param->setParent(*p_Parent);
    }

    return param;
}


void FilmGradePluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
{
    // Source clip only in the filter context
    // Create the mandated source clip
    ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    srcClip->addSupportedComponent(ePixelComponentRGBA);
    srcClip->setTemporalClipAccess(false);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);

    // Create the mandated output clip
    ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->addSupportedComponent(ePixelComponentAlpha);
    dstClip->setSupportsTiles(kSupportsTiles);

    // Make some pages and to things in
    PageParamDescriptor* page = p_Desc.definePageParam("Controls");

    // Make the four component params
    
    GroupParamDescriptor* ecs = p_Desc.defineGroupParam("ExpContSat");
    ecs->setOpen(true);
    ecs->setHint("Exposure Contrast Saturation");
      if (page) {
            page->addChild(*ecs);
    }
    
    GroupParamDescriptor* exp = p_Desc.defineGroupParam("Exposure RGB");
    exp->setOpen(false);
    exp->setHint("Exposure Channels");
    exp->setParent(*ecs);
      if (page) {
            page->addChild(*exp);
    }
    
    {
        RGBParamDescriptor *param = p_Desc.defineRGBParam("expSwatch");
        param->setLabel("Exposure");
        param->setHint("exposure colour wheel");
        param->setDefault(1.0, 1.0, 1.0);
        param->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
        param->setAnimates(true); // can animate
        param->setParent(*exp);
        page->addChild(*param);
    }
    
    DoubleParamDescriptor* param = defineScaleParam(p_Desc, "expR", "Exposure Red", "red offset", exp);
    param->setDefault(0.0);
    param->setRange(-20.0, 20.0);
    param->setIncrement(0.01);
    param->setDisplayRange(-20.0, 20.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "expG", "Exposure Green", "green offset", exp);
    param->setDefault(0.0);
    param->setRange(-20.0, 20.0);
    param->setIncrement(0.01);
    param->setDisplayRange(-20.0, 20.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "expB", "Exposure Blue", "blue offset", exp);
    param->setDefault(0.0);
    param->setRange(-20.0, 20.0);
    param->setIncrement(0.01);
    param->setDisplayRange(-20.0, 20.0);
    page->addChild(*param);
        
    param = defineScaleParam(p_Desc, "exp", "Exposure", "offset", ecs);
    param->setDefault(0.0);
    param->setRange(-20.0, 20.0);
    param->setIncrement(0.01);
    param->setDisplayRange(-20.0, 20.0);
    page->addChild(*param);
    
    GroupParamDescriptor* con = p_Desc.defineGroupParam("Contrast RGB");
    con->setOpen(false);
    con->setHint("Contrast Channels");
    con->setParent(*ecs);
      if (page) {
            page->addChild(*con);
    }
    
    {
        RGBParamDescriptor *param = p_Desc.defineRGBParam("contSwatch");
        param->setLabel("Contrast");
        param->setHint("contrast colour wheel");
        param->setDefault(1.0, 1.0, 1.0);
        param->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
        param->setAnimates(true); // can animate
        param->setParent(*con);
        page->addChild(*param);
    }
    
    param = defineScaleParam(p_Desc, "contR", "Contrast Red", "red contrast", con);
    param->setDefault(1.0);
    param->setRange(0.0, 3.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 3.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "contG", "Contrast Green", "green contrast", con);
    param->setDefault(1.0);
    param->setRange(0.0, 3.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 3.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "contB", "Contrast Blue", "blue contrast", con);
    param->setDefault(1.0);
    param->setRange(0.0, 3.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 3.0);
    page->addChild(*param);
            
    param = defineScaleParam(p_Desc, "cont", "Contrast", "contrast", ecs);
    param->setDefault(1.0);
    param->setRange(0.0, 3.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 3.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "contP", "Contrast Pivot", "contrast pivot point", ecs);
    param->setDefault(445.0);
    param->setRange(0.0, 1023.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1023.0);
    page->addChild(*param);
    
    GroupParamDescriptor* sat = p_Desc.defineGroupParam("Saturation RGB");
    sat->setOpen(false);
    sat->setHint("Contrast Channels");
    sat->setParent(*ecs);
      if (page) {
            page->addChild(*sat);
    }
    
    {
        RGBParamDescriptor *param = p_Desc.defineRGBParam("satSwatch");
        param->setLabel("Saturation");
        param->setHint("saturation colour wheel");
        param->setDefault(1.0, 1.0, 1.0);
        param->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
        param->setAnimates(true); // can animate
        param->setParent(*sat);
        page->addChild(*param);
    }
    
    param = defineScaleParam(p_Desc, "satR", "Saturation Red", "red saturation", sat);
    param->setDefault(1.0);
    param->setRange(0.0, 3.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 3.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "satG", "Saturation Green", "green saturation", sat);
    param->setDefault(1.0);
    param->setRange(0.0, 3.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 3.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "satB", "Saturation Blue", "blue saturation", sat);
    param->setDefault(1.0);
    param->setRange(0.0, 3.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 3.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "sat", "Saturation", "saturation", ecs);
    param->setDefault(1.0);
    param->setRange(0.0, 3.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 3.0);
    page->addChild(*param);
    
    
    GroupParamDescriptor* smh = p_Desc.defineGroupParam("ShadMidHigh");
    smh->setOpen(false);
    smh->setHint("Shadows Midtones Highlights");
      if (page) {
            page->addChild(*smh);
    }
    
    GroupParamDescriptor* shad = p_Desc.defineGroupParam("Shadows RGB");
    shad->setOpen(false);
    shad->setHint("Shadows Channels");
    shad->setParent(*smh);
      if (page) {
            page->addChild(*shad);
    }
    
    {
        RGBParamDescriptor *param = p_Desc.defineRGBParam("shadSwatch");
        param->setLabel("Shadows");
        param->setHint("shadows colour wheel");
        param->setDefault(1.0, 1.0, 1.0);
        param->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
        param->setAnimates(true); // can animate
        param->setParent(*shad);
        page->addChild(*param);
    }
    
    param = defineScaleParam(p_Desc, "shadR", "Shadows Red", "red shadows", shad);
    param->setDefault(0.0);
    param->setRange(-0.5, 0.5);
    param->setIncrement(0.001);
    param->setDisplayRange(-0.5, 0.5);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "shadG", "Shadows Green", "green shadows", shad);
    param->setDefault(0.0);
    param->setRange(-0.5, 0.5);
    param->setIncrement(0.001);
    param->setDisplayRange(-0.5, 0.5);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "shadB", "Shadows Blue", "blue shadows", shad);
    param->setDefault(0.0);
    param->setRange(-0.5, 0.5);
    param->setIncrement(0.001);
    param->setDisplayRange(-0.5, 0.5);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "shad", "Shadows", "shadow region", smh);
    param->setDefault(0.0);
    param->setRange(-0.5, 0.5);
    param->setIncrement(0.001);
    param->setDisplayRange(-0.5, 0.5);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "shadP", "Shadows Pivot", "shadows pivot point", smh);
    param->setDefault(400.0);
    param->setRange(0.0, 1023.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1023.0);
    page->addChild(*param);
    
    GroupParamDescriptor* mid = p_Desc.defineGroupParam("Midtones RGB");
    mid->setOpen(false);
    mid->setHint("Midtones Channels");
    mid->setParent(*smh);
      if (page) {
            page->addChild(*mid);
    }
    
    {
        RGBParamDescriptor *param = p_Desc.defineRGBParam("midSwatch");
        param->setLabel("Midtones");
        param->setHint("midtones colour wheel");
        param->setDefault(1.0, 1.0, 1.0);
        param->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
        param->setAnimates(true); // can animate
        param->setParent(*mid);
        page->addChild(*param);
    }
    
    param = defineScaleParam(p_Desc, "midR", "Midtones Red", "red midtones", mid);
    param->setDefault(0.0);
    param->setRange(-0.5, 0.5);
    param->setIncrement(0.001);
    param->setDisplayRange(-0.5, 0.5);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "midG", "Midtones Green", "green midtones", mid);
    param->setDefault(0.0);
    param->setRange(-0.5, 0.5);
    param->setIncrement(0.001);
    param->setDisplayRange(-0.5, 0.5);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "midB", "Midtones Blue", "blue midtones", mid);
    param->setDefault(0.0);
    param->setRange(-0.5, 0.5);
    param->setIncrement(0.001);
    param->setDisplayRange(-0.5, 0.5);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "mid", "Midtones", "midtones region", smh);
    param->setDefault(0.0);
    param->setRange(-0.5, 0.5);
    param->setIncrement(0.001);
    param->setDisplayRange(-0.5, 0.5);
    page->addChild(*param);
    
    GroupParamDescriptor* high = p_Desc.defineGroupParam("Highlights RGB");
    high->setOpen(false);
    high->setHint("Highlights Channels");
    high->setParent(*smh);
      if (page) {
            page->addChild(*high);
    }
    
    {
        RGBParamDescriptor *param = p_Desc.defineRGBParam("highSwatch");
        param->setLabel("Highlights");
        param->setHint("highlights colour wheel");
        param->setDefault(1.0, 1.0, 1.0);
        param->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
        param->setAnimates(true); // can animate
        param->setParent(*high);
        page->addChild(*param);
    }
    
    param = defineScaleParam(p_Desc, "highR", "Highlights Red", "red highlights", high);
    param->setDefault(0.0);
    param->setRange(-0.5, 0.5);
    param->setIncrement(0.001);
    param->setDisplayRange(-0.5, 0.5);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "highG", "Highlights Green", "green highlights", high);
    param->setDefault(0.0);
    param->setRange(-0.5, 0.5);
    param->setIncrement(0.001);
    param->setDisplayRange(-0.5, 0.5);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "highB", "Highlights Blue", "blue highlights", high);
    param->setDefault(0.0);
    param->setRange(-0.5, 0.5);
    param->setIncrement(0.001);
    param->setDisplayRange(-0.5, 0.5);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "high", "Highlights", "highlight region", smh);
    param->setDefault(0.0);
    param->setRange(-0.5, 0.5);
    param->setIncrement(0.001);
    param->setDisplayRange(-0.5, 0.5);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "highP", "Highlights Pivot", "highlights pivot point", smh);
    param->setDefault(500.0);
    param->setRange(0.0, 1023.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1023.0);
    page->addChild(*param);  
    
    GroupParamDescriptor* adv = p_Desc.defineGroupParam("Advanced");
    adv->setOpen(false);
    adv->setHint("Advanced Controls");
      if (page) {
            page->addChild(*adv);
    }
    
    BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("display");
    boolParam->setDefault(false);
    boolParam->setHint("display curve graph");
    boolParam->setLabel("Display Graph");
    boolParam->setParent(*adv);
    page->addChild(*boolParam);
    
    param = defineScaleParam(p_Desc, "shadPP", "Shadows Pivot", "shadows pivot point", adv);
    param->setDefault(400.0);
    param->setRange(0.0, 1023.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1023.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "highPP", "Highlights Pivot", "highlights pivot point", adv);
    param->setDefault(500.0);
    param->setRange(0.0, 1023.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1023.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "contPP", "Contrast Pivot", "contrast pivot point", adv);
    param->setDefault(445.0);
    param->setRange(0.0, 1023.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1023.0);
    page->addChild(*param);
    
    
}

ImageEffect* FilmGradePluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new FilmGradePlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static FilmGradePluginFactory FilmGradePlugin;
    p_FactoryArray.push_back(&FilmGradePlugin);
}
