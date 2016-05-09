#include "FieldDescriptor.cuh"


map<FieldType,string> InitializeFieldNameStrings()
{
	map<FieldType,string> names;

	names[FieldType::None] = "None"; 
	names[FieldType::Hx]	= "Hx";
	names[FieldType::Hy]	= "Hy";
	names[FieldType::Hz]	= "Hz";
	names[FieldType::Ex]	= "Ex";
	names[FieldType::Ey]	= "Ey";
	names[FieldType::Ez]	= "Ez";
	names[FieldType::Ca]	= "Ca";
	names[FieldType::Cb]	= "Cb";
	names[FieldType::Da]	= "Da";
	names[FieldType::Db]	= "Db";
	names[FieldType::Exy]	= "Exy";
	names[FieldType::Exz]	= "Exz";
	names[FieldType::Eyx]	= "Eyx";
	names[FieldType::Eyz]	= "Eyz";
	names[FieldType::Ezx]	= "Ezx";
	names[FieldType::Ezy]	= "Ezy";
	names[FieldType::Hxy]	= "Hxy";
	names[FieldType::Hxz]	= "Hxz";
	names[FieldType::Hyx]	= "Hyx";
	names[FieldType::Hyz]	= "Hyz";
	names[FieldType::Hzx]	= "Hzx";
	names[FieldType::Hzy]	= "Hzy";

	return names;
}

static map<FieldType,string> fieldNameStrings = InitializeFieldNameStrings();

string to_string(FieldType f)
{
	return fieldNameStrings[f];
}
