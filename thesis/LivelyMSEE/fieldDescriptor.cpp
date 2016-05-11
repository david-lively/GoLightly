struct FieldDescriptor
{
	/// <summary>
	/// describes a split-field boundary region for PML 
	/// </summary>
	struct BoundaryDescriptor
	{
		FieldType Name;
		FieldDirection Direction;

		/// CPU-resident fields
		float *Amp;
		float *Psi;
		float *Decay;

		BoundaryDescriptor *DeviceDescriptor;

		unsigned int AmpDecayLength;

	private:
		CudaHelper *m_cuda;
	};


	float DefaultValue;
	FieldType Name;

	dim3 Size;
	dim3 UpdateRangeStart;
	dim3 UpdateRangeEnd;

	vector<float> HostArray;
	float *DeviceArray;

	DeviceFieldDescriptor *DeviceDescriptor;

	vector<GridBlock> GridBlocks;
	map<FieldType,shared_ptr<BoundaryDescriptor>> Boundaries;
};