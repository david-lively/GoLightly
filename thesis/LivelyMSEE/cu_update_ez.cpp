__global__ void UpdateEz(
	dim3 threadOffset
	)
{
	unsigned int x = threadOffset.x + blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = threadOffset.y + blockIdx.y * blockDim.y + threadIdx.y;

	if (y < 1 || x < 1)
		return;

	float cb = Cb->Data[y * Cb->Size.x + x];

	unsigned int center = y * Ez->Size.x + x;
	float hxBottom = Hx->Data[y * Hx->Size.x + x];
	float hxTop = Hx->Data[(y - 1) * Hx->Size.x + x];
	float dhx = (hxBottom - hxTop);

	float hyRight = Hy->Data[y * Hy->Size.x + x];
	float hyLeft = Hy->Data[y * Hy->Size.x + x - 1];
	float dhy = (hyLeft - hyRight);

	float ezxPsi = 0.f;
	float ezyPsi = 0.f;

	// PML
	if (x < 10 || x > Ez->UpdateRangeEnd.x - 10 || y < 10 || y > Ez->UpdateRangeEnd.y - 10)
	{
		ezyPsi = Ezy->Decay[y] * Ezy->Psi[center] + Ezy->Amp[y] * dhx;
		Ezy->Psi[center] = ezyPsi;
		ezxPsi = Ezx->Decay[x] * Ezx->Psi[center] + Ezx->Amp[x] * dhy;
		Ezx->Psi[center] = ezxPsi;

	}

	Ez->Data[center] = CA * Ez->Data[center] + cb * (dhy - dhx) + cb * (ezxPsi - ezyPsi);
}
