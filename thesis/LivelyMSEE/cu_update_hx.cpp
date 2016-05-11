_global__ void UpdateHx(dim3 threadOffset)
{
	unsigned int x = threadOffset.x + blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = threadOffset.y + blockIdx.y * blockDim.y + threadIdx.y;

	if (y >= Ez->Size.y - 1)
		return;

	unsigned int hxOffset = y * Hx->Size.x + x;
#ifdef USE_MAGNETIC_MATERIALS
	float db = Db->Data[y * Db->Size.x + x];
#else
	const float db = DbDefault;
#endif
	//float ezTop = Ez->Data[y * Ez->Size.x + x];
	//float ezBottom = Ez->Data[(y+1) * Ez->Size.x + x];

	float dEz = Ez->Data[(y + 1) * Ez->Size.x + x] - Ez->Data[y * Ez->Size.x + x];

	float hx = DA * Hx->Data[hxOffset] - db * dEz;

	if (y < 10 || y > Hx->UpdateRangeEnd.y - 10 || x < 10 || x > Hx->UpdateRangeEnd.x - 10)
	{
		/// update boundaries
		float decay = Hxy->Decay[y];
		float amp = Hxy->Amp[y];
		float psi = Hxy->Psi[hxOffset];

		psi = decay * psi + amp * dEz / Configuration->Dx;

		Hxy->Psi[hxOffset] = psi;
		hx = hx - db * Configuration->Dx * psi;
	}

	Hx->Data[hxOffset] = hx;
}

_