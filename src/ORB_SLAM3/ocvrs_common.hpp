#ifndef __OCVRS_COMMON_HPP__
#define __OCVRS_COMMON_HPP__

#if defined(_WIN32) || defined(_WIN64) || defined(__WIN32__) || defined(__TOS_WIN__) || defined(__WINDOWS__) \
 || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
#define OCVRS_TARGET_OS_WINDOWS
#endif

#define CV_COLLECT_IMPL_DATA
// strip dnn experimental ns when generating headers
#ifdef OCVRS_PARSING_HEADERS
	#define CV_DNN_DONT_ADD_EXPERIMENTAL_NS
	#define CV_DNN_DONT_ADD_INLINE_NS
#endif
#include <opencv2/core.hpp>

#define OCVRS_ONLY_DEPENDENT_TYPES
// needed to be able to handle commas in the type name in call to OCVRS_CATCH
#define OCVRS_TYPE(...) __VA_ARGS__

#define CODE_CATCH(return_type, exc_type, code, msg) \
catch (exc_type) { \
	return Err<return_type>(code, msg); \
}

#define OCVRS_CATCH(return_type) \
CODE_CATCH(OCVRS_TYPE(return_type), cv::Exception& e, e.code, e.what()) \
CODE_CATCH(OCVRS_TYPE(return_type), ..., -99999, "unspecified error in OpenCV guts")

#define VEC_CATCH(return_type) \
CODE_CATCH(OCVRS_TYPE(return_type), std::out_of_range, cv::Error::Code::StsOutOfRange, "index out of bounds")

// defined in src/templ.rs
extern "C" void* ocvrs_create_string(const char*);

template<typename T> struct Result {
	int error_code;
	void* error_msg;
	T result;
};

struct Result_void {
	int error_code;
	void* error_msg;
};

template<typename T> inline Result<T> Ok(T result) {
	return Result<T> { 0, 0, result };
}

inline Result_void Ok() {
	return Result_void { 0, 0 };
}

template<typename T> inline T Err(int code, const char* msg) {
	unsigned char ret_buf[sizeof(T)] = {};
	T ret = *reinterpret_cast<T*>(&ret_buf);
	ret.error_code = code;
	ret.error_msg = ocvrs_create_string(msg);
	return ret;
}

#endif
