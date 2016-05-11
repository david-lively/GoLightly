enum class FieldDirection { X,Y,Z };
	
struct DeviceFieldDescriptor
{
	FieldType Name;
	dim3 Size;
	dim3 UpdateRangeStart;
	dim3 UpdateRangeEnd;

	float *Data;
};

