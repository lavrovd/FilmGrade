#ifdef _WIN64
#include <Windows.h>
#else
#include <pthread.h>
#endif
#include <map>
#include <stdio.h>
#include <cmath>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

const char *KernelSource = "\n" \
"#define  BLOCKSIZE 4 \n" \
"__kernel void FilmGradeKernel(                                        \n" \
"   int p_Width,                                                        \n" \
"   int p_Height,                                                       \n" \
"   float p_ExpR,                                                      \n" \
"   float p_ExpG,                                                      \n" \
"   float p_ExpB,                                                      \n" \
"   float p_ContR,                                                      \n" \
"   float p_ContG,                                                      \n" \
"   float p_ContB,                                                      \n" \
"   float p_SatR,                                                      \n" \
"   float p_SatG,                                                      \n" \
"   float p_SatB,                                                      \n" \
"   float p_ShadR,                                                      \n" \
"   float p_ShadG,                                                      \n" \
"   float p_ShadB,                                                      \n" \
"   float p_MidR,                                                      \n" \
"   float p_MidG,                                                      \n" \
"   float p_MidB,                                                      \n" \
"   float p_HighR,                                                      \n" \
"   float p_HighG,                                                      \n" \
"   float p_HighB,                                                      \n" \
"   float p_ShadP,                                                      \n" \
"   float p_HighP,                                                      \n" \
"   float p_ContP,                                                      \n" \
"   float p_Display,                                                      \n" \
"   __global const float* p_Input,                                      \n" \
"   __global float* p_Output)                                           \n" \
"{                                                                      \n" \
"   float SRC[BLOCKSIZE]; \n" \
"   float w_ExpR;                                                      \n" \
"   float w_ExpG;                                                      \n" \
"   float w_ExpB;                                                      \n" \
"   float w_ContR;                                                      \n" \
"   float w_ContG;                                                      \n" \
"   float w_ContB;                                                      \n" \
"   float w_SatR;                                                      \n" \
"   float w_SatG;                                                      \n" \
"   float w_SatB;                                                      \n" \
"   float w_ShadR;                                                      \n" \
"   float w_ShadG;                                                      \n" \
"   float w_ShadB;                                                      \n" \
"   float w_MidR;                                                      \n" \
"   float w_MidG;                                                      \n" \
"   float w_MidB;                                                      \n" \
"   float w_HighR;                                                      \n" \
"   float w_HighG;                                                      \n" \
"   float w_HighB;                                                      \n" \
"   float w_ShadP;                                                      \n" \
"   float w_HighP;                                                      \n" \
"   float w_ContP;                                                      \n" \
"   float w_Display;                                                      \n" \
"	float e;															\n" \
"	float pie;															\n" \
"	float Red;															\n" \
"	float Green;														\n" \
"	float Blue;															\n" \
"	float expR;															\n" \
"	float expG;															\n" \
"	float expB;															\n" \
"	float contR;														\n" \
"	float contG;														\n" \
"	float contB;														\n" \
"	float luma;															\n" \
"	float satR;															\n" \
"	float satG;															\n" \
"	float satB;															\n" \
"	float expr1;														\n" \
"	float expr2;														\n" \
"	float expr3R;														\n" \
"	float expr3G;														\n" \
"	float expr3B;														\n" \
"	float expr4;														\n" \
"	float expr5R;														\n" \
"	float expr5G;														\n" \
"	float expr5B;														\n" \
"	float expr6R;														\n" \
"	float expr6G;														\n" \
"	float expr6B;														\n" \
"	float midR;						    								\n" \
"	float midG;						 								    \n" \
"	float midB;														    \n" \
"	float shadupR1;													    \n" \
"	float shadupR;													    \n" \
"	float shadupG1;													    \n" \
"	float shadupG;													    \n" \
"	float shadupB1;													    \n" \
"	float shadupB;													    \n" \
"	float shaddownR1;												    \n" \
"	float shaddownR;												    \n" \
"	float shaddownG1;												    \n" \
"	float shaddownG;												    \n" \
"	float shaddownB1;												    \n" \
"	float shaddownB;												    \n" \
"	float highupR1;													    \n" \
"	float highupR;													    \n" \
"	float highupG1;													    \n" \
"	float highupG;													    \n" \
"	float highupB1;													    \n" \
"	float highupB;													    \n" \
"	float highdownR1;												    \n" \
"	float highdownR;												    \n" \
"	float highdownG1;												    \n" \
"	float highdownG;												    \n" \
"	float highdownB1;												    \n" \
"	float highdownB;												    \n" \
"   const int x = get_global_id(0);                                     \n" \
"   const int y = get_global_id(1);                                     \n" \
"                                                                       \n" \
"   if ((x < p_Width) && (y < p_Height))                                \n" \
"   {                                                                   \n" \
"       const int index = ((y * p_Width) + x) * BLOCKSIZE;               \n" \
"                                                                       \n" \
"       SRC[0] = p_Input[index + 0] ;    \n" \
"       SRC[1] = p_Input[index + 1] ;    \n" \
"       SRC[2] = p_Input[index + 2] ;    \n" \
"       SRC[3] = p_Input[index + 3] ;    \n" \
"       barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);   \n" \
"       w_ExpR = p_ExpR;                                  \n" \
"       w_ExpG = p_ExpG;                                   \n" \
"       w_ExpB = p_ExpB;                        \n" \
"       w_ContR = p_ContR;					        \n" \
"       w_ContG = p_ContG;                                   \n" \
"       w_ContB = p_ContB;                                   \n" \
"       w_SatR = p_SatR;                                          \n" \
"       w_SatG = p_SatG;                                            \n" \
"       w_SatB = p_SatB;                                            \n" \
"       w_ShadR = p_ShadR;                                          \n" \
"       w_ShadG = p_ShadG;                                           \n" \
"       w_ShadB = p_ShadB;                                           \n" \
"       w_MidR = p_MidR;                                             \n" \
"       w_MidG = p_MidG;                                              \n" \
"       w_MidB = p_MidB;                                              \n" \
"       w_HighR = p_HighR;                                             \n" \
"       w_HighG = p_HighG;                                             \n" \
"       w_HighB = p_HighB;                                             \n" \
"       w_ShadP = p_ShadP;                                             \n" \
"       w_HighP = p_HighP;                                              \n" \
"       w_ContP = p_ContP;                                             \n" \
"       w_Display = p_Display;                                         \n" \
"       barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);   \n" \
"	    e = 2.718281828459045;		\n" \
"       pie = 3.141592653589793;		\n" \
"            	  							\n" \
"       float width = p_Width;     	  							\n" \
"       float height = p_Height;    							\n" \
"	    Red = w_Display != 1.0f ? SRC[0] : x / width;		\n" \
"	    Green = w_Display != 1.0f ? SRC[1] : x / width;	\n" \
"	    Blue = w_Display != 1.0f ? SRC[2] : x / width;		\n" \
"	   															\n" \
"	    expR = Red + w_ExpR/100.0f;		\n" \
"	    expG = Green + w_ExpG/100.0f;		\n" \
"	    expB = Blue + w_ExpB/100.0f;		\n" \
"	   															\n" \
"	    expr1 = (w_ShadP / 2.0f) - (1.0f - w_HighP)/4.0f;		\n" \
"	    expr2 = (1.0f - (1.0f - w_HighP)/2.0f) + (w_ShadP / 4.0f);		\n" \
"	    expr3R = (expR - expr1) / (expr2 - expr1);		\n" \
"	    expr3G = (expG - expr1) / (expr2 - expr1);		\n" \
"	    expr3B = (expB - expr1) / (expr2 - expr1);		\n" \
"	    expr4 =  w_ContP < 0.5f ? 0.5f - (0.5f - w_ContP)/2.0f : 0.5f + (w_ContP - 0.5f)/2.0f;		\n" \
"	    expr5R = expr3R > expr4 ? (expr3R - expr4) / (2.0f - 2.0f*expr4) + 0.5f : expr3R /(2.0f*expr4);	\n" \
"	    expr5G = expr3G > expr4 ? (expr3G - expr4) / (2.0f - 2.0f*expr4) + 0.5f : expr3G /(2.0f*expr4);	\n" \
"	    expr5B = expr3B > expr4 ? (expr3B - expr4) / (2.0f - 2.0f*expr4) + 0.5f : expr3B /(2.0f*expr4);	\n" \
"	    expr6R = (((sin(2.0f * pie * (expr5R -1.0f/4.0f)) + 1.0f) / 20.0f) * w_MidR*4.0f) + expr3R;		\n" \
"	    expr6G = (((sin(2.0f * pie * (expr5G -1.0f/4.0f)) + 1.0f) / 20.0f) * w_MidG*4.0f) + expr3G;		\n" \
"	    expr6B = (((sin(2.0f * pie * (expr5B -1.0f/4.0f)) + 1.0f) / 20.0f) * w_MidB*4.0f) + expr3B;		\n" \
"	    midR = expR >= expr1 && expR <= expr2 ? expr6R * (expr2 - expr1) + expr1 : expR;		\n" \
"	    midG = expG >= expr1 && expG <= expr2 ? expr6G * (expr2 - expr1) + expr1 : expG;		\n" \
"	    midB = expB >= expr1 && expB <= expr2 ? expr6B * (expr2 - expr1) + expr1 : expB;		\n" \
"													\n" \
"		shadupR1 = midR > 0.0f ? 2.0f * (midR/p_ShadP) - log((midR/p_ShadP) * (e * p_ShadR * 2.0f) + 1.0f)/log(e * p_ShadR * 2.0f + 1.0f) : midR;	\n" \
"		shadupR = midR < p_ShadP && p_ShadR > 0.0f ? (shadupR1 + p_ShadR * (1.0f - shadupR1)) * p_ShadP : midR;	\n" \
"		shadupG1 = midG > 0.0f ? 2.0f * (midG/p_ShadP) - log((midG/p_ShadP) * (e * p_ShadG * 2.0f) + 1.0f)/log(e * p_ShadG * 2.0f + 1.0f) : midG;	\n" \
"		shadupG = midG < p_ShadP && p_ShadG > 0.0f ? (shadupG1 + p_ShadG * (1.0f - shadupG1)) * p_ShadP : midG;	\n" \
"		shadupB1 = midB > 0.0f ? 2.0f * (midB/p_ShadP) - log((midB/p_ShadP) * (e * p_ShadB * 2.0f) + 1.0f)/log(e * p_ShadB * 2.0f + 1.0f) : midB;	\n" \
"		shadupB = midB < p_ShadP && p_ShadB > 0.0f ? (shadupB1 + p_ShadB * (1.0f - shadupB1)) * p_ShadP : midB;	\n" \
"	   												\n" \
"		shaddownR1 = shadupR/p_ShadP + p_ShadR*2 * (1.0f - shadupR/p_ShadP);	\n" \
"	    shaddownR = shadupR < p_ShadP && p_ShadR < 0.0f ? (shaddownR1 >= 0.0f ? log(shaddownR1 * (e * p_ShadR * -2.0f) + 1.0f)/log(e * p_ShadR * -2.0f + 1.0f) : shaddownR1) * p_ShadP : shadupR;	\n" \
"	    shaddownG1 = shadupG/p_ShadP + p_ShadG*2 * (1.0f - shadupG/p_ShadP);	\n" \
"	    shaddownG = shadupG < p_ShadP && p_ShadG < 0.0f ? (shaddownG1 >= 0.0f ? log(shaddownG1 * (e * p_ShadG * -2.0f) + 1.0f)/log(e * p_ShadG * -2.0f + 1.0f) : shaddownG1) * p_ShadP : shadupG;	\n" \
"	    shaddownB1 = shadupB/p_ShadP + p_ShadB*2 * (1.0f - shadupB/p_ShadP);	\n" \
"	    shaddownB = shadupB < p_ShadP && p_ShadB < 0.0f ? (shaddownB1 >= 0.0f ? log(shaddownB1 * (e * p_ShadB * -2.0f) + 1.0f)/log(e * p_ShadB * -2.0f + 1.0f) : shaddownB1) * p_ShadP : shadupB;	\n" \
"	   													\n" \
"	    highupR1 = ((shaddownR - w_HighP) / (1.0f - w_HighP)) * (1.0f + (w_HighR * 2.0f));	\n" \
"	    highupR = shaddownR > w_HighP && w_HighP < 1.0f && w_HighR > 0.0f ? (2.0f * highupR1 - log(highupR1 * e * w_HighR + 1.0f)/log(e * w_HighR + 1.0f)) * (1.00001f - w_HighP) + w_HighP : shaddownR;	\n" \
"	    highupG1 = ((shaddownG - w_HighP) / (1.0f - w_HighP)) * (1.0f + (w_HighG * 2.0f));	\n" \
"	    highupG = shaddownG > w_HighP && w_HighP < 1.0f && w_HighG > 0.0f ? (2.0f * highupG1 - log(highupG1 * e * w_HighG + 1.0f)/log(e * w_HighG + 1.0f)) * (1.00001f - w_HighP) + w_HighP : shaddownG;	\n" \
"	    highupB1 = ((shaddownB - w_HighP) / (1.0f - w_HighP)) * (1.0f + (w_HighB * 2.0f));	\n" \
"	    highupB = shaddownB > w_HighP && w_HighP < 1.0f && w_HighB > 0.0f ? (2.0f * highupB1 - log(highupB1 * e * w_HighB + 1.0f)/log(e * w_HighB + 1.0f)) * (1.00001f - w_HighP) + w_HighP : shaddownB;	\n" \
"	   										\n" \
"	    highdownR1 = (highupR - w_HighP) / (1.0f - w_HighP);	\n" \
"	    highdownR = highupR > w_HighP && w_HighP < 1.0f && w_HighR < 0.0f ? log(highdownR1 * (e * w_HighR * -2.0f) + 1.0f)/log(e * w_HighR * -2.0f + 1.0f) * (1.0f + w_HighR) * (1.00001f - w_HighP) + w_HighP : highupR;	\n" \
"	    highdownG1 = (highupG - w_HighP) / (1.0f - w_HighP);	\n" \
"	    highdownG = highupG > w_HighP && w_HighP < 1.0f && w_HighG < 0.0f ? log(highdownG1 * (e * w_HighG * -2.0f) + 1.0f)/log(e * w_HighG * -2.0f + 1.0f) * (1.0f + w_HighG) * (1.00001f - w_HighP) + w_HighP : highupG;	\n" \
"	    highdownB1 = (highupB - w_HighP) / (1.0f - w_HighP);	\n" \
"	    highdownB = highupB > w_HighP && w_HighP < 1.0f && w_HighB < 0.0f ? log(highdownB1 * (e * w_HighB * -2.0f) + 1.0f)/log(e * w_HighB * -2.0f + 1.0f) * (1.0f + w_HighB) * (1.00001f - w_HighP) + w_HighP : highupB;	\n" \
"	   											\n" \
"	    contR = (highdownR - w_ContP) * w_ContR + w_ContP;		\n" \
"	    contG = (highdownG - w_ContP) * w_ContG + w_ContP;		\n" \
"	    contB = (highdownB - w_ContP) * w_ContB + w_ContP;		\n" \
"	   															\n" \
"	    luma = contR * 0.2126f + contG * 0.7152f + contB * 0.0722f;		\n" \
"	    satR = (1.0f - (w_SatR*0.2126f + w_SatG*0.7152f + w_SatB*0.0722f)) * luma + contR * w_SatR;		\n" \
"	    satG = (1.0f - (w_SatR*0.2126f + w_SatG*0.7152f + w_SatB*0.0722f)) * luma + contG * w_SatG;		\n" \
"	    satB = (1.0f - (w_SatR*0.2126f + w_SatG*0.7152f + w_SatB*0.0722f)) * luma + contB * w_SatB;		\n" \
"	   															\n" \
"	    SRC[0] = w_Display != 1.0f ? satR : y / height >= w_ShadP && y / height  <= w_ShadP + 0.005f ? (fmod((float)x, 2.0f) != 0.0f ? 1.0f : 0.0f) : satR >= (y - 5) / height && satR <= (y + 5) / height ? 1.0f : 0.0f;	\n" \
"	    SRC[1] = w_Display != 1.0f ? satG : y / height >= w_HighP && y / height  <= w_HighP + 0.005f ? (fmod((float)x, 2.0f) != 0.0f ? 1.0f : 0.0f) : satG >= (y - 5) / height && satG <= (y + 5) / height ? 1.0f : 0.0f;	\n" \
"	    SRC[2] = w_Display != 1.0f ? satB : y / height >= w_ContP && y / height  <= w_ContP + 0.005f ? (fmod((float)x, 2.0f) != 0.0f ? 1.0f : 0.0f) : satB >= (y - 5) / height && satB <= (y + 5) / height ? 1.0f : 0.0f;	\n" \
"											\n" \
"       p_Output[index + 0] = SRC[0];				\n" \
"       p_Output[index + 1] = SRC[1];				\n" \
"       p_Output[index + 2] = SRC[2];				\n" \
"       p_Output[index + 3] = SRC[3];						\n" \
"       barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);   \n" \
"   }                                                                   \n" \
"}                                                                      \n" \
"\n";

class Locker
{
public:
	Locker()
	{
#ifdef _WIN64
		InitializeCriticalSection(&mutex);
#else
		pthread_mutex_init(&mutex, NULL);
#endif
	}

	~Locker()
	{
#ifdef _WIN64
		DeleteCriticalSection(&mutex);
#else
		pthread_mutex_destroy(&mutex);
#endif
	}

	void Lock()
	{
#ifdef _WIN64
		EnterCriticalSection(&mutex);
#else
		pthread_mutex_lock(&mutex);
#endif
	}

	void Unlock()
	{
#ifdef _WIN64
		LeaveCriticalSection(&mutex);
#else
		pthread_mutex_unlock(&mutex);
#endif
	}

private:
#ifdef _WIN64
	CRITICAL_SECTION mutex;
#else
	pthread_mutex_t mutex;
#endif
};


void CheckError(cl_int p_Error, const char* p_Msg)
{
	if (p_Error != CL_SUCCESS)
	{
		fprintf(stderr, "%s [%d]\n", p_Msg, p_Error);
	}
}

void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_Exp, float* p_Cont, float* p_Sat, 
float* p_Shad, float* p_Mid, float* p_High, float* p_Pivot, float* p_Display, const float* p_Input, float* p_Output)
{
	cl_int error;

	cl_command_queue cmdQ = static_cast<cl_command_queue>(p_CmdQ);

	// store device id and kernel per command queue (required for multi-GPU systems)
	static std::map<cl_command_queue, cl_device_id> deviceIdMap;
	static std::map<cl_command_queue, cl_kernel> kernelMap;

	static Locker locker; // simple lock to control access to the above maps from multiple threads

	locker.Lock();

	// find the device id corresponding to the command queue
	cl_device_id deviceId = NULL;
	if (deviceIdMap.find(cmdQ) == deviceIdMap.end())
	{
		error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_DEVICE, sizeof(cl_device_id), &deviceId, NULL);
		CheckError(error, "Unable to get the device");

		deviceIdMap[cmdQ] = deviceId;
	}
	else
	{
		deviceId = deviceIdMap[cmdQ];
	}

//#define _DEBUG


	// find the program kernel corresponding to the command queue
	cl_kernel kernel = NULL;
	if (kernelMap.find(cmdQ) == kernelMap.end())
	{
		cl_context clContext = NULL;
		error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_CONTEXT, sizeof(cl_context), &clContext, NULL);
		CheckError(error, "Unable to get the context");

		cl_program program = clCreateProgramWithSource(clContext, 1, (const char **)&KernelSource, NULL, &error);
		CheckError(error, "Unable to create program");

		error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
#ifdef _DEBUG
		if (error != CL_SUCCESS)
		{
			char buffer[4096];
			size_t length;
			clGetProgramBuildInfo
				(
				program,
				// valid program object
				deviceId,
				// valid device_id that executable was built
				CL_PROGRAM_BUILD_LOG,
				// indicate to retrieve build log
				sizeof(buffer),
				// size of the buffer to write log to
				buffer,
				// the actual buffer to write log to
				&length);
			// the actual size in bytes of data copied to buffer
			FILE * pFile;
			pFile = fopen("e:\\FilmGradeKernel_OPENCL.txt", "w");
			if (pFile != NULL)
			{
				fprintf(pFile, "%s\n", buffer);
				//fprintf(pFile, "%s [%lu]\n", "localWorkSize 0 =", szWorkSize);
			}
			fclose(pFile);
		}
#else
		CheckError(error, "Unable to build program");
#endif

		kernel = clCreateKernel(program, "FilmGradeKernel", &error);
		CheckError(error, "Unable to create kernel");

		kernelMap[cmdQ] = kernel;
	}
	else
	{
		kernel = kernelMap[cmdQ];
	}

	locker.Unlock();

    int count = 0;
    error  = clSetKernelArg(kernel, count++, sizeof(int), &p_Width);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &p_Height);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Exp[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Exp[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Exp[2]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Cont[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Cont[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Cont[2]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Sat[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Sat[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Sat[2]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Shad[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Shad[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Shad[2]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Mid[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Mid[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Mid[2]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_High[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_High[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_High[2]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Pivot[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Pivot[1]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Pivot[2]);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Display[0]);
    error |= clSetKernelArg(kernel, count++, sizeof(cl_mem), &p_Input);
    error |= clSetKernelArg(kernel, count++, sizeof(cl_mem), &p_Output);
    CheckError(error, "Unable to set kernel arguments");

    size_t localWorkSize[2], globalWorkSize[2];
    clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), localWorkSize, NULL);
    localWorkSize[1] = 1;
    globalWorkSize[0] = ((p_Width + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
    globalWorkSize[1] = p_Height;

    clEnqueueNDRangeKernel(cmdQ, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}
