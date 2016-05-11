__global__ void UpdateHy(dim3 threadOffset)
{
	unsigned int x = threadOffset.x + blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = threadOffset.y + blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= Ez->Size.x - 1)
		return;

	unsigned int hyOffset = y * Hy->Size.x + x;

#ifdef USE_MAGNETIC_MATERIALS
	float db = Db->Data[y * Db->Size.x + x];
#else
	const float db = DbDefault;
#endif

	float ezLeft = Ez->Data[y * Ez->Size.x + x];
	float ezRight = Ez->Data[y * Ez->Size.x + x + 1];

	float dEz = ezRight - ezLeft;
	float hy = DA * Hy->Data[hyOffset] - db * (ezRight - ezLeft);

	if (x < 10 || y < 10 || x > Hy->UpdateRangeEnd.x - 10 || y > Hy->UpdateRangeEnd.y - 10)
	{

		float psi = Hyx->Psi[hyOffset];
		float decay = Hyx->Decay[x];
		float amp = Hyx->Amp[x];

		psi = decay * psi + amp * dEz / Configuration->Dx;

		hy = hy - db * Configuration->Dx * psi;

		Hyx->Psi[hyOffset] = psi;
	}

	Hy->Data[hyOffset] = hy;
}