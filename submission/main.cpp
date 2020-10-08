#include "assignment.hpp"

// ******* Function Member Implementation *******

// ***** World function members *****
Colour World::max_to_one(const Colour& c) const {
	float max_value = std::max(c.r, std::max(c.g, c.b));
	if (max_value > 1.0)
		return (c / max_value);
	else
		return (c);
}

Colour World::clamp(const Colour& raw_color) const {
	Colour c(raw_color);
	if (raw_color.r > 1.0 || raw_color.g > 1.0 || raw_color.b > 1.0) {
		c = { 1, 0, 0 };
	}
	return c;
}

ShadeRec World::hit_objects(const Ray& ray) {
	ShadeRec trace_data{};
	trace_data.world = shared_from_this();
	trace_data.t = std::numeric_limits<float>::max();

	bool hit{};

	for (auto obj : this->scene)
	{
		hit |= obj->hit(ray, trace_data);
	}

	if (hit) {
		trace_data.hit_an_object = true;
	}

	return trace_data;
}

Shape::Shape() : mColour{ NULL }
{}

void Shape::setColour(Colour const& col)
{
	mColour = col;
}

void Shape::setMaterial(std::shared_ptr<Material> const& material)
{
	mMaterial = material;
}

Colour Shape::getColour() const
{
	return mColour;
}

std::shared_ptr<Material> Shape::getMaterial() const
{
	return mMaterial;
}

atlas::math::Point Shape::sample() {
	return atlas::math::Point();
}

atlas::math::Normal Shape::get_normal() {
	return atlas::math::Normal();
}

atlas::math::Normal Shape::get_normal([[maybe_unused]] atlas::math::Point const& p) {
	return atlas::math::Normal();
}

float Shape::pdf([[maybe_unused]] ShadeRec const& sr) {
	return 1.0f;
}

// ***** Sampler function members *****
Sampler::Sampler(int numSamples, int numSets) :
	mNumSamples{ numSamples }, mNumSets{ numSets }, mCount{ 0 }, mJump{ 0 }
{
	mSamples.reserve(mNumSets* mNumSamples);
	setupShuffledIndeces();
}

int Sampler::getNumSamples() const
{
	return mNumSamples;
}

void Sampler::setupShuffledIndeces()
{
	mShuffledIndeces.reserve(mNumSamples * mNumSets);
	std::vector<int> indices;

	std::random_device d;
	std::mt19937 generator(d());

	for (int j = 0; j < mNumSamples; ++j)
	{
		indices.push_back(j);
	}

	for (int p = 0; p < mNumSets; ++p)
	{
		std::shuffle(indices.begin(), indices.end(), generator);

		for (int j = 0; j < mNumSamples; ++j)
		{
			mShuffledIndeces.push_back(indices[j]);
		}
	}
}

atlas::math::Point Sampler::sampleUnitSquare()
{
	if (mCount % mNumSamples == 0)
	{
		atlas::math::Random<int> engine;
		mJump = (engine.getRandomMax() % mNumSets) * mNumSamples;
	}

	return mSamples[mJump + mShuffledIndeces[mJump + mCount++ % mNumSamples]];
}

void Sampler::map_samples_to_hemisphere(const float e) {
	size_t size = mSamples.size();
	mHemisphereSamples.reserve(mNumSamples * mNumSets);

	for (int j = 0; j < size; j++) {
		float cos_phi = cos(2.0f * (float)M_PI * mSamples[j].x);
		float sin_phi = sin(2.0f * (float)M_PI * mSamples[j].x);
		float cos_theta = pow((1.0f - mSamples[j].y), 1.0f / (e + 1.0f));
		float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
		float pu = sin_theta * cos_phi;
		float pv = sin_theta * sin_phi;
		float pw = cos_theta;

		mHemisphereSamples.push_back(atlas::math::Point(pu, pv, pw));

	}

}

atlas::math::Point Sampler::sampleHemisphere() {
	if (mCount % mNumSamples == 0)
	{
		atlas::math::Random<int> engine;
		mJump = (engine.getRandomMax() % mNumSets) * mNumSamples;
	}

	return mHemisphereSamples[mJump + mShuffledIndeces[mJump + mCount++ % mNumSamples]];
}


// ***** Light function members *****

Light::Light()
	: mColour{ 1,1,1 },
	mRadiance{ 0.2f },
	mShadows{ true }
{}

bool Light::castsShadows() const {
	return mShadows;
}

void Light::setShadows(bool t) {
	mShadows = t;
}

void Light::scaleRadiance([[maybe_unused]] float b)
{
	mRadiance = b;
}

void Light::setColour([[maybe_unused]] Colour const& c)
{
	mColour = c;
}

Colour Light::L([[maybe_unused]] ShadeRec& sr)
{
	return Colour{ 0.0f };
}

float Light::G([[maybe_unused]] ShadeRec const& sr) const {
	return 1.0f;
}

float Light::pdf([[maybe_unused]] ShadeRec const& sr) const {
	return 1.0f;
}

Ambient::Ambient()
	: Light(),
	mColour{ 1,1,1 },
	mRadiance{ 0.2f }
{};

atlas::math::Vector Ambient::getDirection([[maybe_unused]] ShadeRec& sr) {
	return { 0,0,0 }; // returns 0 because not called anywhere
}

Colour Ambient::L([[maybe_unused]] ShadeRec& sr) {
	return mRadiance * mColour;
}

bool Ambient::in_shadow([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray, [[maybe_unused]] ShadeRec& sr) const {
	return(false);
}



AmbientOccluder::AmbientOccluder()
	: Light(),
	mMinAmount{ NULL },
	mU{ NULL },
	mV{ NULL },
	mW{ NULL }
{}

AmbientOccluder::AmbientOccluder(Colour min)
	: Light(),
	mMinAmount{ min }
{}

void AmbientOccluder::set_sampler(std::shared_ptr<Sampler> shared_ptr)
{
	// might not need these with a shared pointer
	//if (mSamplerPtr) {
	//    delete mSamplerPtr;
	//    mSamplerPtr = NULL;
	//}

	mSamplerPtr = shared_ptr;
	mSamplerPtr->map_samples_to_hemisphere(1);
}

atlas::math::Vector AmbientOccluder::getDirection([[maybe_unused]] ShadeRec& sr) {
	atlas::math::Point sp = mSamplerPtr->sampleHemisphere();
	return (sp.x * mU + sp.y * mV + sp.z * mW);
}

void AmbientOccluder::set_min_amount(Colour min) {
	mMinAmount = min;
}

bool AmbientOccluder::in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const {
	float t;
	size_t num_objects = sr.world->scene.size();

	for (int j = 0; j < num_objects; j++)
		if (sr.world->scene[j]->intersectRay(ray, t)) {
			return true;
		}

	return false;
}

Colour AmbientOccluder::L(ShadeRec& sr) {
	mW = sr.normal;

	mV = glm::cross(mW, atlas::math::Vector(0.0072f, 1.0f, 0.0034f));
	mV = glm::normalize(mV);
	mU = glm::cross(mW, mV);

	Ray shadow_ray;
	shadow_ray.o = sr.hit_point;
	shadow_ray.d = getDirection(sr);
	if (in_shadow(shadow_ray, sr))
		return(mMinAmount * mRadiance * mColour);
	else
		return(mRadiance * mColour);


}

PointL::PointL(atlas::math::Point l)
	: Light(),
	mColour{ 1,1,1 },
	mRadiance{ 1.0f },
	mLocation{ l }
{}

atlas::math::Vector PointL::getDirection([[maybe_unused]] ShadeRec& sr) {
	return glm::normalize(mLocation - sr.hit_point);
}

void PointL::scaleRadiance(float b) {
	mRadiance = b;
}

Colour PointL::L([[maybe_unused]] ShadeRec& sr) {
	return mRadiance * mColour;
}


bool PointL::in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const {
	float t;
	size_t num_objects = sr.world->scene.size();
	float d = glm::distance(mLocation, ray.o);

	for (int j = 0; j < num_objects; j++) {
		if (sr.world->scene[j]->intersectRay(ray, t) && t < d)
			return (true);
	}

	return(false);
}

Directional::Directional(atlas::math::Vector dir)
	: Light(),
	mColour{ 1,1,1 },
	mRadiance{ 1.0f },
	mDir{ dir }
{}

atlas::math::Vector Directional::getDirection([[maybe_unused]] ShadeRec& sr) {
	return mDir;
}

Colour Directional::L([[maybe_unused]] ShadeRec& sr) {
	return mRadiance * mColour;
}

bool Directional::in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const {
	float t;
	size_t num_objects = sr.world->scene.size();

	for (int j = 0; j < num_objects; j++) {
		if (sr.world->scene[j]->intersectRay(ray, t))
			return (true);
	}

	return(false);
}

AreaLight::AreaLight()
	: Light()
{}

void AreaLight::set_object(std::shared_ptr<Shape> shape_ptr) {
	mShapePtr = shape_ptr;
	if (mShapePtr) {
		mMaterialPtr = shape_ptr->getMaterial();
	}
}

atlas::math::Vector AreaLight::getDirection(ShadeRec& sr) {
	mSamplePoint = mShapePtr->sample();
	mLightNormal = mShapePtr->get_normal(mSamplePoint);
	mWi = mSamplePoint - sr.hit_point;
	mWi = glm::normalize(mWi);

	return mWi;
}
bool AreaLight::in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const {
	float t;
	float ts = glm::dot((mSamplePoint - ray.o), ray.d);

	for (int j = 0; j < sr.world->scene.size(); j++) {
		if (sr.world->scene[j]->intersectRay(ray, t) && t < ts)
			return true;
	}
	return false;
}
Colour AreaLight::L(ShadeRec& sr) {
	float ndotd = glm::dot(-mLightNormal, mWi); //weird stuff going on with the normal here, it should be negative but flipped the sign to make it work

	if (ndotd > 0.0f) {
		return mMaterialPtr->get_Le(sr);
	}
	else
		return { 0, 0, 0 };
}
float AreaLight::G(ShadeRec const& sr) const {
	float ndotd = glm::dot(-mLightNormal, mWi);
	float d2 = glm::distance2(mSamplePoint, sr.hit_point);

	return (ndotd / d2);

}
float AreaLight::pdf(ShadeRec const& sr) const {
	return mShapePtr->pdf(sr);
}



// ***** Sphere function members *****
Sphere::Sphere(atlas::math::Point center, float radius) :
	mCentre{ center }, mRadius{ radius }, mRadiusSqr{ radius * radius }
{}

bool Sphere::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
	ShadeRec& sr) const
{
	atlas::math::Vector tmp = ray.o - mCentre;
	float t{ std::numeric_limits<float>::max() };
	bool intersect{ intersectRay(ray, t) };

	// update ShadeRec info about new closest hit
	if (intersect && t < sr.t)
	{
		sr.normal = (tmp + t * ray.d) / mRadius;
		sr.ray = ray;
		sr.color = mColour;
		sr.t = t;
		sr.material = mMaterial;
		sr.hit_point = ray.o + ray.d * t;
	}

	return intersect;
}

bool Sphere::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
	float& tMin) const
{
	const auto tmp{ ray.o - mCentre };
	const auto a{ glm::dot(ray.d, ray.d) };
	const auto b{ 2.0f * glm::dot(ray.d, tmp) };
	const auto c{ glm::dot(tmp, tmp) - mRadiusSqr };
	const auto disc{ (b * b) - (4.0f * a * c) };

	if (atlas::core::geq(disc, 0.0f))
	{
		const float kEpsilon{ 0.01f };
		const float e{ std::sqrt(disc) };
		const float denom{ 2.0f * a };

		// Look at the negative root first
		float t = (-b - e) / denom;
		if (atlas::core::geq(t, kEpsilon))
		{
			tMin = t;
			return true;
		}

		// Now the positive root
		t = (-b + e);
		if (atlas::core::geq(t, kEpsilon))
		{
			tMin = t;
			return true;
		}
	}

	return false;
}


// ***** Triangle function members *****

Triangle::Triangle(atlas::math::Point p0, atlas::math::Point p1, atlas::math::Point p2, atlas::math::Vector n) :
	mP0{ p0 }, mP1{ p1 }, mP2{ p2 }, mNorm{ n }
{}

bool Triangle::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
	ShadeRec& sr) const
{
	float t{ std::numeric_limits<float>::max() };
	bool intersect{ intersectRay(ray, t) };

	// update ShadeRec info about new closest hit
	if (intersect && t < sr.t)
	{
		sr.normal = mNorm;
		sr.ray = ray;
		sr.color = mColour;
		sr.t = t;
		sr.material = mMaterial;
		sr.hit_point = ray.o + ray.d * t;
	}

	return intersect;
}


// from textbook
bool Triangle::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
	float& tMin) const
{
	const float kEpsilon{ 0.001f };
	float a = mP0.x - mP1.x, b = mP0.x - mP2.x, c = ray.d.x, d = mP0.x - ray.o.x;
	float e = mP0.y - mP1.y, f = mP0.y - mP2.y, g = ray.d.y, h = mP0.y - ray.o.y;
	float i = mP0.z - mP1.z, j = mP0.z - mP2.z, k = ray.d.z, l = mP0.z - ray.o.z;

	float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
	float q = g * i - e * k, s = e * j - f * i;

	float inv_denom = 1.0f / (a * m + b * q + c * s);

	float e1 = d * m - b * n - c * p;
	float beta = e1 * inv_denom;

	if (beta < 0.0)
		return (false);

	float r = e * l - h * i;
	float e2 = a * n + d * q + c * r;
	float gamma = e2 * inv_denom;

	if (gamma < 0.0f)
		return (false);

	if (beta + gamma > 1.0f)
		return (false);

	float e3 = a * p - b * r + d * s;
	float t = e3 * inv_denom;

	if (t < kEpsilon)
		return false;

	tMin = t;

	return true;
}

// ***** Plane function members *****

Plane::Plane(atlas::math::Point po, atlas::math::Vector norm) :
	mPoint{ po }, mNormal{ norm }
{}

bool Plane::hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const
{
	float t{ std::numeric_limits<float>::max() };
	bool intersect{ intersectRay(ray,t) };

	if (intersect && t < sr.t)
	{
		sr.normal = mNormal;
		sr.ray = ray;
		sr.color = mColour;
		sr.t = t;
		sr.material = mMaterial;
		sr.hit_point = ray.o + ray.d * t;
	}

	return intersect;

}

bool Plane::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
{

	const float kEpsilon{ 0.01f };

	float t = glm::dot((mPoint - ray.o), mNormal) / glm::dot(ray.d, mNormal);

	if (atlas::core::geq(t, kEpsilon)) {
		tMin = t;
		return true;
	}

	return false;

}

// ***** Rectangle function members *****

Rectangle::Rectangle(atlas::math::Point p0, atlas::math::Vector a, atlas::math::Vector b)
	: Shape(),
	mP0{ p0 },
	mA{ a },
	mB{ b },
	mNormal{ glm::normalize(glm::cross(mA, mB)) },
	mInvArea{ 1.0f / (glm::length(mA) * glm::length(mB)) }
{}

bool Rectangle::hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const {
	float t{ std::numeric_limits<float>::max() };
	bool intersect{ intersectRay(ray,t) };

	if (intersect && t < sr.t)
	{
		sr.normal = mNormal;
		sr.ray = ray;
		sr.color = mColour;
		sr.t = t;
		sr.material = mMaterial;
		sr.hit_point = ray.o + ray.d * t;
	}

	return intersect;
}

void Rectangle::set_sampler(std::shared_ptr<Sampler> sampler_ptr) {
	mSamplerPtr = sampler_ptr;
}

atlas::math::Point Rectangle::sample() {
	atlas::math::Point2 sample_point = mSamplerPtr->sampleUnitSquare();
	return (mP0 + sample_point.x * mA + sample_point.y * mB);
}

float Rectangle::pdf([[maybe_unused]] ShadeRec& sr) {
	return mInvArea;
}

atlas::math::Normal Rectangle::get_normal() {
	return mNormal;
}

atlas::math::Normal Rectangle::get_normal([[maybe_unused]] atlas::math::Point const& p) {
	return mNormal;
}

bool Rectangle::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const
{
	const float kEpsilon{ 0.01f };
	float t = glm::dot((mP0 - ray.o), mNormal) / glm::dot(ray.d, mNormal);
	if (t <= kEpsilon)
		return false;

	atlas::math::Point p = ray.o + t * ray.d;
	atlas::math::Vector d = p - mP0;

	float ddota = glm::dot(d, mA);

	if (ddota < 0.0f || ddota >(glm::length(mA) * glm::length(mA)))
		return(false);

	float ddotb = glm::dot(d, mB);

	if (ddotb < 0.0f || ddotb >(glm::length(mB) * glm::length(mB)))
		return(false);

	tMin = t;

	return true;

}


// ***** Box function members *****

Box::Box(atlas::math::Point p1, atlas::math::Point p2)
	: Shape(),
	mPoint1{ p1 },
	mPoint2{ p2 }
{}

bool Box::hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const
{
	float t{ std::numeric_limits<float>::max() };

	float ox = ray.o.x; float oy = ray.o.y; float oz = ray.o.z;
	float dx = ray.d.x; float dy = ray.d.y; float dz = ray.d.z;

	const float kEpsilon{ 0.01f };

	float tx_min, ty_min, tz_min;
	float tx_max, ty_max, tz_max;

	float a = 1.0f / dx;
	if (a >= 0) {
		tx_min = (mPoint1.x - ox) * a;
		tx_max = (mPoint2.x - ox) * a;
	}
	else {
		tx_min = (mPoint2.x - ox) * a;
		tx_max = (mPoint1.x - ox) * a;
	}

	float b = 1.0f / dy;
	if (b >= 0) {
		ty_min = (mPoint1.y - oy) * b;
		ty_max = (mPoint2.y - oy) * b;
	}
	else {
		ty_min = (mPoint2.y - oy) * b;
		ty_max = (mPoint1.y - oy) * b;
	}

	float c = 1.0f / dz;
	if (c >= 0) {
		tz_min = (mPoint1.z - oz) * c;
		tz_max = (mPoint2.z - oz) * c;
	}
	else {
		tz_min = (mPoint2.z - oz) * c;
		tz_max = (mPoint1.z - oz) * c;
	}

	float t0, t1;
	int face_in, face_out;

	if (tx_min > ty_min) {
		t0 = tx_min;
		face_in = (a >= 0.0) ? 0 : 3;
	}
	else {
		t0 = ty_min;
		face_in = (b >= 0.0) ? 1 : 4;
	}

	if (tz_min > t0) {
		t0 = tz_min;
		face_in = (c >= 0.0) ? 2 : 5;
	}

	if (tx_max < ty_max) {
		t1 = tx_max;
		face_out = (a >= 0.0) ? 3 : 0;
	}
	else {
		t1 = ty_max;
		face_out = (b >= 0.0) ? 4 : 1;
	}

	if (tz_max < t1) {
		t1 = tz_max;
		face_out = (c >= 0.0) ? 5 : 2;
	}

	if (t0 < t1 && t1 > kEpsilon) {
		if (t0 > kEpsilon) {
			t = t0;
			if (t < sr.t) {
				sr.normal = get_normal(face_in);
				sr.ray = ray;
				sr.color = mColour;
				sr.t = t;
				sr.material = mMaterial;
				sr.hit_point = ray.o + ray.d * t;
				return true;
			}
		}
		else {
			t = t1;
			if (t < sr.t) {
				sr.normal = get_normal(face_out);
				sr.ray = ray;
				sr.color = mColour;
				sr.t = t;
				sr.material = mMaterial;
				sr.hit_point = ray.o + ray.d * t;
				return true;
			}
		}

	}
	return false;

}

bool Box::intersectRay([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray, [[maybe_unused]] float& tMin) const
{
	float ox = ray.o.x; float oy = ray.o.y; float oz = ray.o.z;
	float dx = ray.d.x; float dy = ray.d.y; float dz = ray.d.z;

	const float kepsilon{ 0.01f };

	float tx_min, ty_min, tz_min;
	float tx_max, ty_max, tz_max;

	float a = 1.0f / dx;
	if (a >= 0) {
		tx_min = (mPoint1.x - ox) * a;
		tx_max = (mPoint2.x - ox) * a;
	}
	else {
		tx_min = (mPoint2.x - ox) * a;
		tx_max = (mPoint1.x - ox) * a;
	}

	float b = 1.0f / dy;
	if (b >= 0) {
		ty_min = (mPoint1.y - oy) * b;
		ty_max = (mPoint2.y - oy) * b;
	}
	else {
		ty_min = (mPoint2.y - oy) * b;
		ty_max = (mPoint1.y - oy) * b;
	}

	float c = 1.0f / dz;
	if (c >= 0) {
		tz_min = (mPoint1.z - oz) * c;
		tz_max = (mPoint2.z - oz) * c;
	}
	else {
		tz_min = (mPoint2.z - oz) * c;
		tz_max = (mPoint1.z - oz) * c;
	}

	float t0, t1;
	int face_in, face_out;

	if (tx_min > ty_min) {
		t0 = tx_min;
		face_in = (a >= 0.0) ? 0 : 3;
	}
	else {
		t0 = ty_min;
		face_in = (b >= 0.0) ? 1 : 4;
	}

	if (tz_min > t0) {
		t0 = tz_min;
		face_in = (c >= 0.0) ? 2 : 5;
	}

	if (tx_max < ty_max) {
		t1 = tx_max;
		face_out = (a >= 0.0) ? 3 : 0;
	}
	else {
		t1 = ty_max;
		face_out = (b >= 0.0) ? 4 : 1;
	}

	if (tz_max < t1) {
		t1 = tz_max;
		face_out = (b >= 0.0) ? 5 : 2;
	}

	if (t0 < t1 && t1 > kepsilon) {
		if (t0 > kepsilon) {
			tMin = t0;
		}
		else {
			tMin = t1;
		}

		return true;
	}

	return false;


}

atlas::math::Vector Box::get_normal(const int face_hit) const {
	switch (face_hit) {
	case 0: return { -1, 0, 0 };
	case 1: return { 0, -1, 0 };
	case 2: return { 0,0,-1 };
	case 3: return { 1,0,0 };
	case 4: return { 0,1,0 };
	case 5: return { 0,0,1 };
	default: return { 0,0,0 };
	}
}

// ***** Material function members *****

Colour Material::area_light_shade([[maybe_unused]] ShadeRec& sr) {
	return Colour{ 0,0,0 };
}

Colour Material::get_Le([[maybe_unused]] ShadeRec& sr) const {
	return Colour{ 0,0,0 };
}

Matte::Matte(std::shared_ptr<Lambertian> ambient_brdf_ptr, std::shared_ptr<Lambertian> diffuse_brdf_ptr)
	: Material(),
	mAmbientBRDF{ ambient_brdf_ptr },
	mDiffuseBRDF{ diffuse_brdf_ptr }
{}

Colour Matte::shade(ShadeRec& sr) {
	atlas::math::Vector wo = -sr.ray.d;
	Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);

	//from textbook
	for (int j = 0; j < sr.world->lights.size(); j++) {
		atlas::math::Vector wi = sr.world->lights[j]->getDirection(sr);
		float ndotwi = glm::dot(sr.normal, wi);

		if (ndotwi > 0.0) {
			bool in_shadow = false;

			if (sr.world->lights[j]->castsShadows()) {
				atlas::math::Ray<atlas::math::Vector> shadowRay;
				shadowRay.o = sr.hit_point;
				shadowRay.d = wi;
				in_shadow = sr.world->lights[j]->in_shadow(shadowRay, sr);
			}

			if (!in_shadow) {
				L += mDiffuseBRDF->fn(sr, wo, wi) * sr.world->lights[j]->L(sr) * ndotwi;
			}
		}
	}

	return (L);

	// add the mAmbientBRDF and the mdiffuseBRDF to get the color
}

void Matte::set_cd(Colour c) {
	mAmbientBRDF->set_cd(c);
	mDiffuseBRDF->set_cd(c);
}

void Matte::set_ka(float k) {
	mAmbientBRDF->set_kd(k);
}

void Matte::set_kd(float k) {
	mDiffuseBRDF->set_kd(k);
}

Colour Matte::area_light_shade(ShadeRec& sr) {
	atlas::math::Vector wo = -sr.ray.d;
	Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);

	//from textbook
	for (int j = 0; j < sr.world->lights.size(); j++) {
		atlas::math::Vector wi = sr.world->lights[j]->getDirection(sr);
		float ndotwi = glm::dot(sr.normal, wi);

		if (ndotwi > 0.0) {
			bool in_shadow = false;

			if (sr.world->lights[j]->castsShadows()) {
				atlas::math::Ray<atlas::math::Vector> shadowRay;
				shadowRay.o = sr.hit_point;
				shadowRay.d = wi;
				in_shadow = sr.world->lights[j]->in_shadow(shadowRay, sr);
			}

			if (!in_shadow) {
				L += mDiffuseBRDF->fn(sr, wo, wi) * sr.world->lights[j]->L(sr) * sr.world->lights[j]->G(sr) * ndotwi / sr.world->lights[j]->pdf(sr);
			}
		}
	}

	return (L);
}

SV_Matte::SV_Matte(std::shared_ptr<SV_Lambertian> ambient_brdf_ptr, std::shared_ptr<SV_Lambertian> diffuse_brdf_ptr)
	: Material(),
	mAmbientBRDF{ ambient_brdf_ptr },
	mDiffuseBRDF{ diffuse_brdf_ptr }
{}

void SV_Matte::set_cd(std::shared_ptr<Texture> t_ptr) {
	mAmbientBRDF->set_cd(t_ptr);
	mDiffuseBRDF->set_cd(t_ptr);
}

void SV_Matte::set_ka(float k) {
	mAmbientBRDF->set_kd(k);
}

void SV_Matte::set_kd(float k) {
	mDiffuseBRDF->set_kd(k);
}

Colour SV_Matte::shade(ShadeRec& sr) {
	atlas::math::Vector wo = -sr.ray.d;
	Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);

	//from textbook
	for (int j = 0; j < sr.world->lights.size(); j++) {
		atlas::math::Vector wi = sr.world->lights[j]->getDirection(sr);
		float ndotwi = glm::dot(sr.normal, wi);

		if (ndotwi > 0.0) {
			bool in_shadow = false;

			if (sr.world->lights[j]->castsShadows()) {
				atlas::math::Ray<atlas::math::Vector> shadowRay;
				shadowRay.o = sr.hit_point;
				shadowRay.d = wi;
				in_shadow = sr.world->lights[j]->in_shadow(shadowRay, sr);
			}

			if (!in_shadow) {
				L += mDiffuseBRDF->fn(sr, wo, wi) * sr.world->lights[j]->L(sr) * ndotwi;
			}
		}
	}

	return (L);

	// add the mAmbientBRDF and the mdiffuseBRDF to get the color
}

Colour SV_Matte::area_light_shade(ShadeRec& sr) {
	atlas::math::Vector wo = -sr.ray.d;
	Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);

	//from textbook
	for (int j = 0; j < sr.world->lights.size(); j++) {
		atlas::math::Vector wi = sr.world->lights[j]->getDirection(sr);
		float ndotwi = glm::dot(sr.normal, wi);

		if (ndotwi > 0.0) {
			bool in_shadow = false;

			if (sr.world->lights[j]->castsShadows()) {
				atlas::math::Ray<atlas::math::Vector> shadowRay;
				shadowRay.o = sr.hit_point;
				shadowRay.d = wi;
				in_shadow = sr.world->lights[j]->in_shadow(shadowRay, sr);
			}

			if (!in_shadow) {
				L += mDiffuseBRDF->fn(sr, wo, wi) * sr.world->lights[j]->L(sr) * sr.world->lights[j]->G(sr) * ndotwi / sr.world->lights[j]->pdf(sr);
			}
		}
	}

	return (L);
}

Phong::Phong(std::shared_ptr<Lambertian> ambient_brdf_ptr, std::shared_ptr<Lambertian> diffuse_brdf_ptr, std::shared_ptr<GlossySpecular> glossy_specular_brdf_ptr)
	: Material(),
	mAmbientBRDF{ ambient_brdf_ptr },
	mDiffuseBRDF{ diffuse_brdf_ptr },
	mSpecularBRDF{ glossy_specular_brdf_ptr }
{}

void Phong::set_ka(const float k) {
	mAmbientBRDF->set_kd(k);
}

void Phong::set_kd(const float k) {
	mDiffuseBRDF->set_kd(k);
}

void Phong::set_ks(const float k) {
	mSpecularBRDF->set_ks(k);
}

void Phong::set_e(const float e) {
	mSpecularBRDF->set_e(e);
}

void Phong::set_cd(Colour c) {
	mAmbientBRDF->set_cd(c);
	mDiffuseBRDF->set_cd(c);
}

Colour Phong::shade(ShadeRec& sr) {
	atlas::math::Vector wo = -sr.ray.d;
	Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);

	//from textbook
	for (int j = 0; j < sr.world->lights.size(); j++) {
		atlas::math::Vector wi = sr.world->lights[j]->getDirection(sr);
		float ndotwi = glm::dot(sr.normal, wi);

		if (ndotwi > 0.0) {
			bool in_shadow = false;

			if (sr.world->lights[j]->castsShadows()) {
				atlas::math::Ray<atlas::math::Vector> shadowRay;
				shadowRay.o = sr.hit_point;
				shadowRay.d = wi;
				in_shadow = sr.world->lights[j]->in_shadow(shadowRay, sr);
			}

			if (!in_shadow) {
				L += (mDiffuseBRDF->fn(sr, wo, wi) + mSpecularBRDF->fn(sr, wo, wi)) * sr.world->lights[j]->L(sr) * ndotwi;
			}
		}
	}

	return (L);
}

Colour Phong::area_light_shade(ShadeRec& sr) {
	atlas::math::Vector wo = -sr.ray.d;
	Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);

	//from textbook
	for (int j = 0; j < sr.world->lights.size(); j++) {
		atlas::math::Vector wi = sr.world->lights[j]->getDirection(sr);
		float ndotwi = glm::dot(sr.normal, wi);

		if (ndotwi > 0.0) {
			bool in_shadow = false;

			if (sr.world->lights[j]->castsShadows()) {
				atlas::math::Ray<atlas::math::Vector> shadowRay;
				shadowRay.o = sr.hit_point;
				shadowRay.d = wi;
				in_shadow = sr.world->lights[j]->in_shadow(shadowRay, sr);
			}

			if (!in_shadow) {
				L += (mDiffuseBRDF->fn(sr, wo, wi) + mSpecularBRDF->fn(sr, wo, wi)) * sr.world->lights[j]->L(sr)
					* sr.world->lights[j]->G(sr) * ndotwi / sr.world->lights[j]->pdf(sr);
			}
		}
	}

	return (L);
}


Reflective::Reflective(std::shared_ptr<Lambertian> ambient_brdf_ptr,
	std::shared_ptr<Lambertian> diffuse_brdf_ptr,
	std::shared_ptr<GlossySpecular> glossy_specular_brdf_ptr,
	std::shared_ptr<PerfectSpecular> perfect_specular_brdf_ptr)
	: Phong(ambient_brdf_ptr, diffuse_brdf_ptr, glossy_specular_brdf_ptr),
	mPerfectSpecularBRDF{ perfect_specular_brdf_ptr }
{}

void Reflective::set_kr(const float k) {
	mPerfectSpecularBRDF->set_kr(k);
}

void Reflective::set_cr(Colour const& c) {
	mPerfectSpecularBRDF->set_cr(c);
}

Colour Reflective::shade(ShadeRec& sr) {
	Colour L{ Phong::shade(sr) };

	atlas::math::Vector wo = -sr.ray.d;
	atlas::math::Vector wi;

	Colour fr = mPerfectSpecularBRDF->sample_f(sr, wo, wi);
	Ray reflected_ray{ sr.hit_point, wi };

	L += fr * sr.world->tracer->trace_ray(reflected_ray, sr.depth + 1) * glm::dot(sr.normal, wi);

	return (L);



}

GlossyReflector::GlossyReflector(std::shared_ptr<Lambertian> ambient_brdf_ptr,
	std::shared_ptr<Lambertian> diffuse_brdf_ptr,
	std::shared_ptr<GlossySpecular> glossy_specular_brdf_ptr1,
	std::shared_ptr<GlossySpecular> glossy_specular_brdf_ptr2)
	: Phong(ambient_brdf_ptr, diffuse_brdf_ptr, glossy_specular_brdf_ptr1),
	mGlossySpecularBRDF{ glossy_specular_brdf_ptr2 }
{};

void GlossyReflector::set_samples(const int num_samples, const float exp) {
	mGlossySpecularBRDF->set_samples(num_samples, exp);
}

void GlossyReflector::set_kr(const float k) {
	mGlossySpecularBRDF->set_ks(k);
}

void GlossyReflector::set_exponent(const float exp) {
	mGlossySpecularBRDF->set_e(exp);
}

void GlossyReflector::set_cr(Colour c) {
	mGlossySpecularBRDF->set_cs(c);
}

Colour GlossyReflector::shade(ShadeRec& sr) {
	Colour L(Phong::shade(sr));
	atlas::math::Vector wo{ -sr.ray.d };
	atlas::math::Vector wi;
	float pdf;
	Colour fr(mGlossySpecularBRDF->sample_f(sr, wo, wi, pdf));
	Ray reflected_ray(sr.hit_point, wi);

	L += fr * sr.world->tracer->trace_ray(reflected_ray, sr.depth + 1) * (glm::dot(sr.normal, wi)) / pdf;

	return L;
}

Emissive::Emissive()
	: Material(),
	mLs{ NULL },
	mCe{ NULL }
{};

Emissive::Emissive(float l, Colour c)
	: Material(), mLs{ l }, mCe{ c }
{};

void Emissive::scale_radiance(float ls) {
	mLs = ls;
}

void Emissive::set_ce(Colour c) {
	mCe = c;
}

Colour Emissive::get_Le([[maybe_unused]] ShadeRec& sr) const {
	return mLs * mCe;
}

Colour Emissive::shade([[maybe_unused]] ShadeRec& sr) {
	return Colour{ 1,1,1 };
}

Colour Emissive::area_light_shade([[maybe_unused]] ShadeRec& sr) {
	if (glm::dot(-sr.normal, sr.ray.d) > 0.0f)
		return mLs * mCe;
	else
		return Colour{ 0, 0 , 0 };
}



//BRDFs

Lambertian::Lambertian()
	: BRDF(),
	mDiffuseColour{ 1,1,1 },
	mDiffuseReflection{ 1.0f }
{}

Lambertian::Lambertian(float k, Colour c)
	: BRDF(),
	mDiffuseColour{ c },
	mDiffuseReflection{ k }
{}

void Lambertian::set_kd(float k) {
	mDiffuseReflection = k;
}

void Lambertian::set_cd(Colour c) {
	mDiffuseColour = c;
}

Colour Lambertian::fn([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector const& incoming) const
{
	return (mDiffuseReflection * sr.color * (float)M_1_PI);
}

Colour Lambertian::rho([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected) const
{
	return (mDiffuseReflection * sr.color);
}

SV_Lambertian::SV_Lambertian()
	: BRDF(),
	mCd{ NULL },
	mKd{ NULL }
{}

void SV_Lambertian::set_kd(float k) {
	mKd = k;
}

void SV_Lambertian::set_cd(std::shared_ptr<Texture> t) {
	mCd = t;
}

Colour SV_Lambertian::rho([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected) const {
	return mKd * mCd->get_colour(sr);
}

Colour SV_Lambertian::fn([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector const& incoming) const {
	return mKd * mCd->get_colour(sr) * (float)M_1_PI;
}

//GlossySpecular

GlossySpecular::GlossySpecular()
	: BRDF(),
	mKs{ NULL },
	mPhongExponent{ NULL },
	mCs{ NULL }
{}

GlossySpecular::GlossySpecular(float k, float e, Colour c)
	: BRDF(),
	mKs{ k },
	mPhongExponent{ e },
	mCs{ c }
{}

void GlossySpecular::set_samples(const int num_samples, const float exp) {
	mSamplerPtr = std::make_shared<Jitter>(num_samples, 83);
	mSamplerPtr->map_samples_to_hemisphere(exp);
}

void GlossySpecular::set_sampler(std::shared_ptr<Sampler> sampler_ptr) {
	mSamplerPtr = sampler_ptr;
}

void GlossySpecular::set_cs(Colour c) {
	mCs = c;
}

void GlossySpecular::set_e(float e) {
	mPhongExponent = e;
}

void GlossySpecular::set_ks(float k) {
	mKs = k;
}

Colour GlossySpecular::sample_f(ShadeRec const& sr, atlas::math::Vector const& wo, atlas::math::Vector& wi, float& pdf) const {
	float ndotwo = glm::dot(sr.normal, wo);
	atlas::math::Vector r = -wo + 2.0f * sr.normal * ndotwo;

	atlas::math::Vector w = r;
	atlas::math::Vector u = glm::cross(atlas::math::Vector{ 0.00424f, 1.0f, 0.00764f }, w);
	u = glm::normalize(u);
	atlas::math::Vector v = glm::cross(u, v);

	atlas::math::Point sp = mSamplerPtr->sampleHemisphere();
	wi = sp.x * u + sp.y * v + sp.z * w;

	if (glm::dot(sr.normal, wi) < 0.0f) {
		wi = -sp.x * u - sp.y * v + sp.z * w;
	}

	float phong_lobe = pow(glm::dot(r, wi), mPhongExponent);
	pdf = phong_lobe * glm::dot(sr.normal, wi);

	return mKs * mCs * phong_lobe;

}

Colour GlossySpecular::fn([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector const& incoming) const
{
	Colour L;
	float ndotwi = glm::dot(sr.normal, incoming);
	atlas::math::Vector r{ -incoming + 2.0f * sr.normal * ndotwi };
	r = glm::normalize(r);
	float rdotwo = glm::dot(r, reflected);

	L = sr.color * mKs * pow(std::max(rdotwo, 0.0f), mPhongExponent);

	return L;
}

Colour GlossySpecular::rho([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected) const
{
	return { 0,0,0 };
}

PerfectSpecular::PerfectSpecular()
	: BRDF(),
	mKr{ 0.75f },
	mCr{ 1.0f, 1.0f, 1.0f }
{}

PerfectSpecular::PerfectSpecular(float k, Colour const& c)
	: BRDF(),
	mKr{ k },
	mCr{ c }
{}


Colour PerfectSpecular::fn([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector const& incoming) const
{
	return Colour{ 0,0,0 };
}

Colour PerfectSpecular::rho([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected) const
{
	return Colour{ 0,0,0 };
}

Colour PerfectSpecular::sample_f(ShadeRec const& sr, atlas::math::Vector const& reflected, atlas::math::Vector& incoming) const {
	float ndotwo = glm::dot(sr.normal, reflected);
	incoming = -reflected + 2.0f * ndotwo * sr.normal;

	return (mKr * mCr / glm::dot(sr.normal, incoming));
}

void PerfectSpecular::set_kr(const float k) {
	mKr = k;
}

void PerfectSpecular::set_cr(Colour const& c) {
	mCr = c;
}

// Textures

Texture::Texture()
{}

Checker3D::Checker3D()
	: Texture(),
	mColour1{ NULL },
	mColour2{ NULL },
	mSize{ NULL }
{}

Checker3D::Checker3D(Colour c1, Colour c2, float s)
	: Texture(),
	mColour1{ c1 },
	mColour2{ c2 },
	mSize{ s }
{}

void Checker3D::set_size(float s) {
	mSize = s;
}

void Checker3D::set_colour1(Colour c1) {
	mColour1 = c1;
}

void Checker3D::set_colour2(Colour c2) {
	mColour2 = c2;
}

Colour Checker3D::get_colour(ShadeRec const& sr) const {
	float x = sr.hit_point.x;
	float y = sr.hit_point.y;
	float z = sr.hit_point.z;

	if (((int)floor(x / mSize) + (int)floor(y / mSize) + (int)floor(z / mSize)) % 2 == 0)
		return mColour1;
	else
		return mColour2;
}
// ***** Regular function members *****
Regular::Regular(int numSamples, int numSets) : Sampler{ numSamples, numSets }
{
	generateSamples();
}

void Regular::generateSamples()
{
	int n = static_cast<int>(glm::sqrt(static_cast<float>(mNumSamples)));

	for (int j = 0; j < mNumSets; ++j)
	{
		for (int p = 0; p < n; ++p)
		{
			for (int q = 0; q < n; ++q)
			{
				mSamples.push_back(
					atlas::math::Point{ (q + 0.5f) / n, (p + 0.5f) / n, 0.0f });
			}
		}
	}
}

// ***** Regular function members *****
Random::Random(int numSamples, int numSets) : Sampler{ numSamples, numSets }
{
	generateSamples();
}

void Random::generateSamples()
{
	atlas::math::Random<float> engine;
	for (int p = 0; p < mNumSets; ++p)
	{
		for (int q = 0; q < mNumSamples; ++q)
		{
			mSamples.push_back(atlas::math::Point{
				engine.getRandomOne(), engine.getRandomOne(), 0.0f });
		}
	}
}

// ***** Jitter function members *****
Jitter::Jitter(int numSamples, int numSets) : Sampler{ numSamples, numSets }
{
	generateSamples();
}

void Jitter::generateSamples()
{
	int n = (int)sqrt(mNumSamples);
	atlas::math::Random<float> engine;
	for (int p = 0; p < mNumSets; ++p)
	{
		for (int q = 0; q < n; ++q)
		{
			for (int r = 0; r < n; ++r) {
				mSamples.push_back(atlas::math::Point{
					(r + engine.getRandomOne()) / n, (q + engine.getRandomOne()) / n, 0.0f
					});
			}
		}
	}
}

// ***** Camera function members *****
Camera::Camera() :
	mEye{ 0.0f, 0.0f, 500.0f },
	mLookAt{ 0.0f },
	mUp{ 0.0f, 1.0f, 0.0f },
	mU{ 1.0f, 0.0f, 0.0f },
	mV{ 0.0f, 1.0f, 0.0f },
	mW{ 0.0f, 0.0f, 1.0f }
{}

void Camera::setEye(atlas::math::Point const& eye)
{
	mEye = eye;
}

void Camera::setLookAt(atlas::math::Point const& lookAt)
{
	mLookAt = lookAt;
}

void Camera::setUpVector(atlas::math::Vector const& up)
{
	mUp = up;
}

void Camera::computeUVW()
{
	mW = (mEye - mLookAt) / glm::length(mEye - mLookAt);
	mU = glm::cross(mUp, mW) / glm::length(glm::cross(mUp, mW));
	mV = glm::cross(mW, mU);
}

Pinhole::Pinhole()
	: Camera(),
	d{ 0.0f },
	zoom{ 0.0f }
{}

void Pinhole::set_view_distance(float vd)
{
	d = vd;
}

void Pinhole::renderScene(std::shared_ptr<World> world) const {

	using atlas::math::Point;
	using atlas::math::Ray;
	using atlas::math::Vector;

	Point samplePoint{}, pixelPoint{};
	Ray<atlas::math::Vector> ray;

	ray.o = mEye;

	float avg{ 1.0f / world->sampler->getNumSamples() };

	for (int r{ 0 }; r < world->height; ++r)
	{
		for (int c{ 0 }; c < world->width; ++c)
		{
			Colour pixelAverage{ 0, 0, 0 };

			for (int j = 0; j < world->sampler->getNumSamples(); ++j)
			{

				samplePoint = world->sampler->sampleUnitSquare();
				pixelPoint.x = c - 0.5f * world->width + samplePoint.x;
				pixelPoint.y = r - 0.5f * world->height + samplePoint.y;
				ray.d = ray_direction(pixelPoint);

				ShadeRec trace_data{ world->hit_objects(ray) };

				pixelAverage += trace_data.world->tracer->trace_ray(ray);

			}

			Colour avgColor = { pixelAverage.r * avg,
									pixelAverage.g * avg,
									pixelAverage.b * avg };


			avgColor = world->max_to_one(avgColor);

			world->image.push_back(avgColor);
		}
	}
}



atlas::math::Vector Pinhole::ray_direction(const atlas::math::Point2& p) const {
	atlas::math::Vector dir = p.x * mU + p.y * mV - d * mW;
	return glm::normalize(dir);
}

Fisheye::Fisheye()
	: Camera(),
	mPsiMax{ 0.0f }
{}

void Fisheye::set_psi_max(float psi) {
	mPsiMax = psi;
}

void Fisheye::renderScene(std::shared_ptr<World> world) const {

	atlas::math::Ray<atlas::math::Vector> ray;
	atlas::math::Point2 samplePoint;
	atlas::math::Point2 pixelPoint;
	float r_squared;

	ray.o = mEye;

	float avg{ 1.0f / world->sampler->getNumSamples() };

	for (int r = 0; r < world->height; r++) {
		for (int c = 0; c < world->width; c++) {
			Colour pixelAverage{ 0, 0, 0 };

			for (int j = 0; j < world->sampler->getNumSamples(); ++j)
			{
				ShadeRec trace_data{};
				trace_data.world = world;
				trace_data.t = std::numeric_limits<float>::max();
				samplePoint = world->sampler->sampleUnitSquare();
				pixelPoint.x = c - 0.5f * world->width + samplePoint.x;
				pixelPoint.y = r - 0.5f * world->height + samplePoint.y;
				ray.d = ray_direction(pixelPoint, world->width, world->height, r_squared);


				bool hit{};

				for (auto obj : world->scene)
				{
					hit |= obj->hit(ray, trace_data);
				}

				if (hit && r_squared <= 1.0)
				{
					pixelAverage += trace_data.material->shade(trace_data);
				}
			}

			Colour avgColor = { pixelAverage.r * avg,
									pixelAverage.g * avg,
									pixelAverage.b * avg };


			avgColor = world->max_to_one(avgColor);

			world->image.push_back(avgColor);

		}
	}

}

atlas::math::Vector Fisheye::ray_direction(const atlas::math::Point2& pixelPoint, const size_t width,
	const size_t height, float& r_squared) const {

	atlas::math::Point2 pn{ 2.0f / width * pixelPoint.x, 2.0f / height * pixelPoint.y };
	r_squared = pn.x * pn.x + pn.y * pn.y;

	if (r_squared <= 1.0f) {
		float r = sqrt(r_squared);
		float psi = r * mPsiMax * ((float)M_PI / 180.0f);
		float sin_psi = sin(psi);
		float cos_psi = cos(psi);
		float sin_alpha = pn.y / r;
		float cos_alpha = pn.x / r;

		atlas::math::Vector dir = sin_psi * cos_alpha * mU + sin_psi * sin_alpha * mV - cos_psi * mW;
		return dir;
	}

	return (atlas::math::Vector{ 0,0,0 });
}


Tracer::Tracer(std::shared_ptr<World> world_ptr)
	: mWorld{ world_ptr }
{}

RayCast::RayCast(std::shared_ptr<World> world_ptr) :
	Tracer(world_ptr)
{}

Whitted::Whitted(std::shared_ptr<World> world_ptr) :
	Tracer(world_ptr)
{}

AreaLighting::AreaLighting(std::shared_ptr<World> world_ptr) :
	Tracer(world_ptr)
{}

Colour RayCast::trace_ray(const Ray& ray) const {
	ShadeRec sr(mWorld->hit_objects(ray));

	if (sr.hit_an_object) {
		return sr.material->shade(sr);
	}
	else {
		return mWorld->background;
	}
}

Colour RayCast::trace_ray(const Ray ray, [[maybe_unused]] const int depth) const {
	ShadeRec sr(mWorld->hit_objects(ray));

	if (sr.hit_an_object) {
		return sr.material->shade(sr);
	}
	else {
		return mWorld->background;
	}
}


Colour Whitted::trace_ray(const Ray& ray) const {
	ShadeRec sr(mWorld->hit_objects(ray));

	if (sr.hit_an_object) {
		return sr.material->shade(sr);
	}
	else {
		return mWorld->background;
	}
}

Colour Whitted::trace_ray(const Ray ray, const int depth) const {
	if (depth > mWorld->max_depth)
		return { 0, 0, 0 };
	else {
		ShadeRec sr(mWorld->hit_objects(ray));

		if (sr.hit_an_object) {
			sr.depth = depth;
			sr.ray = ray;
			if (sr.material == nullptr) {
				return mWorld->background;
			}
			return sr.material->shade(sr);
		}
		else {
			return mWorld->background;
		}
	}
}

Colour AreaLighting::trace_ray(const Ray& ray) const {
	ShadeRec sr(mWorld->hit_objects(ray));

	if (sr.hit_an_object) {
		sr.ray = ray;
		return sr.material->area_light_shade(sr);
	}
	else {
		return mWorld->background;
	}
}

Colour AreaLighting::trace_ray(const Ray ray, [[maybe_unused]] const int depth) const {

	ShadeRec sr(mWorld->hit_objects(ray));

	if (sr.hit_an_object) {
		sr.ray = ray;
		return sr.material->area_light_shade(sr);
	}
	else {
		return mWorld->background;
	}

}

Transparent::Transparent(std::shared_ptr<Lambertian> ambient_brdf_ptr, std::shared_ptr<Lambertian> diffuse_brdf_ptr,
	std::shared_ptr<GlossySpecular> glossy_specular_brdf_ptr, std::shared_ptr<PerfectSpecular> perfect_specular_brdf_ptr,
	std::shared_ptr<PerfectTransmitter> perfect_transmitter_btdf_ptr)
	: Phong(ambient_brdf_ptr, diffuse_brdf_ptr, glossy_specular_brdf_ptr),
	mPerfectSpecularBRDF{ perfect_specular_brdf_ptr },
	mPerfectTransmitterBTDF{ perfect_transmitter_btdf_ptr }
{}

void Transparent::set_kt(float k) {
	mPerfectTransmitterBTDF->set_kt(k);
}

void Transparent::set_ior(float i) {
	mPerfectTransmitterBTDF->set_ior(i);
}

void Transparent::set_kr(float k) {
	mPerfectSpecularBRDF->set_kr(k);
}

void Transparent::set_cr(Colour c) {
	mPerfectSpecularBRDF->set_cr(c);
}

Colour Transparent::shade(ShadeRec& sr) {
	Colour L{ Phong::shade(sr) };

	atlas::math::Vector wo = -sr.ray.d;
	atlas::math::Vector wi;
	Colour fr = mPerfectSpecularBRDF->sample_f(sr, wo, wi);
	Ray reflected_ray{ sr.hit_point, wi };

	if (mPerfectTransmitterBTDF->tir(sr))
		L += sr.world->tracer->trace_ray(reflected_ray, sr.depth + 1);
	else {
		atlas::math::Vector wt;
		Colour ft = mPerfectTransmitterBTDF->sample_f(sr, wo, wt);
		Ray transmitted_ray(sr.hit_point, wt);

		L += fr * sr.world->tracer->trace_ray(reflected_ray, sr.depth + 1) * fabs(glm::dot(sr.normal, wi));
		L += ft * sr.world->tracer->trace_ray(transmitted_ray, sr.depth + 1) * fabs(glm::dot(sr.normal, wt));
	}

	return L;
}

void PerfectTransmitter::set_kt(float k) {
	mKt = k;
}

void PerfectTransmitter::set_ior(float i) {
	mIor = i;
}

Colour PerfectTransmitter::fn([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector const& incoming) const
{
	return { 0, 0, 0 };
}

Colour PerfectTransmitter::rho([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected) const
{
	return Colour{ 0,0,0 };
}

Colour PerfectTransmitter::sample_f(ShadeRec const& sr, atlas::math::Vector const& wo, atlas::math::Vector& wt) const
{
	atlas::math::Normal n{ sr.normal };
	float cos_thetai = glm::dot(n, wo);
	float eta = mIor;

	if (cos_thetai < 0.0f) {
		cos_thetai = -cos_thetai;
		n = -n;
		eta = 1.0f / eta;
	}

	float temp = (1.0f - (1.0f - cos_thetai * cos_thetai) / (eta * eta));
	float cos_theta2 = std::sqrt(temp);
	wt = -wo / eta - (cos_theta2 - cos_thetai / eta) * n; //glm::dot here?

	Colour white{ 1.0f, 1.0f, 1.0f };

	return (mKt / (eta * eta) * white / std::fabs(glm::dot(sr.normal, wt)));
}

bool PerfectTransmitter::tir(ShadeRec const& sr) const {
	atlas::math::Vector wo{ -sr.ray.d };
	float cos_thetai = glm::dot(sr.normal, wo);
	float eta = mIor;

	if (cos_thetai < 0.0f)
		eta = 1.0f / eta;

	return (1.0f - (1.0f - cos_thetai * cos_thetai) / (eta * eta)) < 0.0f;
}





// ******* Driver Code *******

void create_scene(std::shared_ptr<World> world) {
	using atlas::math::Point;
	using atlas::math::Ray;
	using atlas::math::Vector;

	world->width = 1000;
	world->height = 1000;
	world->background = { 0, 0, 0 };
	world->sampler = std::make_shared<Jitter>(256, 83); //first number here is the number of samples
	world->max_depth = 2;

	std::shared_ptr<Ambient> ambient_ptr(new Ambient());

	//ambient occluder
	std::shared_ptr<AmbientOccluder> occluder_ptr(new AmbientOccluder());
	occluder_ptr->scaleRadiance(0.5f);
	occluder_ptr->setColour({ 1.0f, 1.0f, 1.0f });
	occluder_ptr->set_min_amount({ 0.1f, 0.1f, 0.1f });
	std::shared_ptr<Sampler> occluder_sampler = std::make_shared<Jitter>(256, 83);
	occluder_ptr->set_sampler(occluder_sampler);
	world->ambient = occluder_ptr;

	std::shared_ptr<Tracer> tracer_ptr(new Whitted(world));
	world->tracer = tracer_ptr;

	//checker
	std::shared_ptr<Checker3D> checker3d_ptr(new Checker3D());
	checker3d_ptr->set_size(20.0f);
	checker3d_ptr->set_colour1({ 0, 0, 0 });
	checker3d_ptr->set_colour2({ 1, 1, 1 });

	//sv matte
	std::shared_ptr<SV_Lambertian> sv_ambient_brdf_ptr(new SV_Lambertian());
	std::shared_ptr<SV_Lambertian> sv_diffuse_brdf_ptr(new SV_Lambertian());
	std::shared_ptr<SV_Matte> sv_matte_ptr(new SV_Matte(sv_ambient_brdf_ptr, sv_diffuse_brdf_ptr));
	sv_matte_ptr->set_ka(2.0f);
	sv_matte_ptr->set_kd(2.0f);
	sv_matte_ptr->set_cd(checker3d_ptr);

	//matte
	std::shared_ptr<Lambertian> ambient_brdf_ptr(new Lambertian());
	std::shared_ptr<Lambertian> diffuse_brdf_ptr(new Lambertian());
	std::shared_ptr<Matte> matte_ptr(new Matte(ambient_brdf_ptr, diffuse_brdf_ptr));
	matte_ptr->set_ka(1.5f);
	matte_ptr->set_kd(1.5f);

	//phong
	std::shared_ptr<GlossySpecular> glossy_specular_brdf_ptr(new GlossySpecular());
	std::shared_ptr<Phong> phong_ptr(new Phong(ambient_brdf_ptr, diffuse_brdf_ptr, glossy_specular_brdf_ptr));

	//reflective
	std::shared_ptr<Lambertian> ambient_brdf_ptr2(new Lambertian());
	std::shared_ptr<Lambertian> diffuse_brdf_ptr2(new Lambertian());
	std::shared_ptr<GlossySpecular> glossy_specular_brdf_ptr2(new GlossySpecular());
	std::shared_ptr<PerfectSpecular> perfect_specular_brdf_ptr(new PerfectSpecular());
	std::shared_ptr<Reflective> reflective_ptr(new Reflective(ambient_brdf_ptr2, diffuse_brdf_ptr2, glossy_specular_brdf_ptr2, perfect_specular_brdf_ptr));

	//ambient_ptr->scaleRadiance(0.25f);
	reflective_ptr->set_ka(0.25f);
	reflective_ptr->set_kd(1.0f);
	reflective_ptr->set_ks(0.15f);
	reflective_ptr->set_e(10000.0f);
	reflective_ptr->set_kr(0.75f);

	//glossyreflective
	std::shared_ptr<Lambertian> ambient_brdf_ptr3(new Lambertian());
	std::shared_ptr<Lambertian> diffuse_brdf_ptr3(new Lambertian());
	std::shared_ptr<GlossySpecular> glossy_specular_brdf_ptr3(new GlossySpecular());
	std::shared_ptr<GlossySpecular> glossy_specular_brdf_ptr4(new GlossySpecular());
	std::shared_ptr<GlossyReflector> glossy_reflective_ptr(new GlossyReflector(ambient_brdf_ptr3, diffuse_brdf_ptr3, glossy_specular_brdf_ptr3, glossy_specular_brdf_ptr4));

	int num_samples = 256;
	float exp = 100;
	glossy_reflective_ptr->set_samples(num_samples, exp);
	glossy_reflective_ptr->set_ka(0.0f);
	glossy_reflective_ptr->set_kd(0.0f);
	glossy_reflective_ptr->set_ks(0.0f);
	glossy_reflective_ptr->set_e(exp);
	glossy_reflective_ptr->set_kr(0.9f);
	glossy_reflective_ptr->set_cd({ 0, 0.75, 0.75 });
	glossy_reflective_ptr->set_exponent(exp);
	glossy_reflective_ptr->set_cr({ 0, 0.75, 0.75 });

	//transparent
	std::shared_ptr<Lambertian> ambient_brdf_ptr4(new Lambertian());
	ambient_brdf_ptr4->set_kd(0.1f);
	std::shared_ptr<Lambertian> diffuse_brdf_ptr4(new Lambertian());
	diffuse_brdf_ptr4->set_kd(0.6f);
	std::shared_ptr<GlossySpecular> glossy_specular_brdf_ptr5(new GlossySpecular());
	std::shared_ptr<PerfectSpecular> perfect_specular_brdf_ptr2(new PerfectSpecular());
	std::shared_ptr<PerfectTransmitter> perfect_transmitter_btdf_ptr(new PerfectTransmitter());
	std::shared_ptr<Transparent> glass_ptr(new Transparent(ambient_brdf_ptr4, diffuse_brdf_ptr4, glossy_specular_brdf_ptr5, perfect_specular_brdf_ptr2, perfect_transmitter_btdf_ptr));

	glass_ptr->set_ks(0.6f);
	glass_ptr->set_e(1000.0f);
	glass_ptr->set_ior(1.01f);
	glass_ptr->set_kr(0.3f);
	glass_ptr->set_kt(1.0f);


	// right sphere
	std::shared_ptr<Sphere> right_sphere_ptr = std::make_shared<Sphere>(atlas::math::Point{ 156, -50, -870 }, 120.0f);
	right_sphere_ptr->setMaterial(reflective_ptr);
	world->scene.push_back(right_sphere_ptr);

	//left sphere
	std::shared_ptr<Sphere> left_sphere_ptr = std::make_shared<Sphere>(atlas::math::Point{ -156, -50, -870 }, 120.0f);
	left_sphere_ptr->setMaterial(glossy_reflective_ptr);
	left_sphere_ptr->setColour({ 1,1,1 });
	world->scene.push_back(left_sphere_ptr);

	//middle sphere
	std::shared_ptr<Sphere> middle_sphere_ptr = std::make_shared<Sphere>(atlas::math::Point{ 0, -50, -700 }, 100.0f);
	middle_sphere_ptr->setMaterial(glass_ptr);
	middle_sphere_ptr->setColour({ 1,1,1 });
	world->scene.push_back(middle_sphere_ptr);

	//floor
	std::shared_ptr<Plane> floor_plane_ptr = std::make_shared<Plane>(atlas::math::Point{ 0, 100, 0 }, atlas::math::Vector{ 0,-1,0 });
	floor_plane_ptr->setMaterial(sv_matte_ptr);
	world->scene.push_back(floor_plane_ptr);


	//back plane
	std::shared_ptr<Plane> back_plane_ptr = std::make_shared<Plane>(atlas::math::Point{ 0, 0, -1000 }, atlas::math::Vector{ 0,0, 1 });
	back_plane_ptr->setColour({ 1, 1, 1 });
	back_plane_ptr->setMaterial(matte_ptr);
	world->scene.push_back(back_plane_ptr);

	//behind plane
	std::shared_ptr<Plane> behind_plane_ptr = std::make_shared<Plane>(atlas::math::Point{ 0, 0, 1000 }, atlas::math::Vector{ 0,0, 1 });
	behind_plane_ptr->setColour({ 1, 1, 1, });
	behind_plane_ptr->setMaterial(matte_ptr);
	world->scene.push_back(behind_plane_ptr);

	//left plane
	std::shared_ptr<Plane> left_plane_ptr = std::make_shared<Plane>(atlas::math::Point{ -400, 0, 0 }, atlas::math::Vector{ 1 ,0, 0 });
	left_plane_ptr->setColour({ 1, 0.4, 0, });
	left_plane_ptr->setMaterial(matte_ptr);
	world->scene.push_back(left_plane_ptr);

	//right plane
	std::shared_ptr<Plane> right_plane_ptr = std::make_shared<Plane>(atlas::math::Point{ 400, 0, 0 }, atlas::math::Vector{ -1 ,0, 0 });
	right_plane_ptr->setColour({ 0.5,0.55, 0.97, });
	right_plane_ptr->setMaterial(matte_ptr);
	world->scene.push_back(right_plane_ptr);

	//top plane
	std::shared_ptr<Plane> top_plane_ptr = std::make_shared<Plane>(atlas::math::Point{ 0, -500, 0 }, atlas::math::Vector{ 0 , 1, 0 });
	top_plane_ptr->setColour({ 1, 1, 1, });
	top_plane_ptr->setMaterial(matte_ptr);
	world->scene.push_back(top_plane_ptr);

	//area light

	std::shared_ptr<Emissive> emissive_ptr(new Emissive());
	emissive_ptr->scale_radiance(3.0f);
	emissive_ptr->set_ce({ 1,1,1 });

	std::shared_ptr<Rectangle> rectangle_ptr = std::make_shared<Rectangle>(atlas::math::Point{ -200, -499, -800 }, atlas::math::Vector{ 0, 0, 200 }, atlas::math::Vector{ 400, 0 , 0 });
	rectangle_ptr->setMaterial(emissive_ptr);
	rectangle_ptr->set_sampler(world->sampler);
	rectangle_ptr->setColour({ 1,1,1 });
	world->scene.push_back(rectangle_ptr);

	std::shared_ptr<AreaLight> area_light_ptr = std::make_shared<AreaLight>();
	area_light_ptr->set_object(rectangle_ptr);
	world->lights.push_back(area_light_ptr);

	std::shared_ptr<PointL> point_light_ptr = std::make_shared<PointL>(atlas::math::Point{ 0 , -200, -300});
	point_light_ptr->setShadows(false);
	point_light_ptr->scaleRadiance(0.01f);
	world->lights.push_back(point_light_ptr);
}


void render_pinhole() {

	std::shared_ptr<World> world{ std::make_shared<World>() };

	create_scene(world);

	std::shared_ptr<Pinhole> pinhole_ptr(new Pinhole());
	pinhole_ptr->setEye({ 0, -200, 0 });
	pinhole_ptr->setLookAt({ 0,-100,-700 });
	pinhole_ptr->set_view_distance(700);
	pinhole_ptr->computeUVW();
	pinhole_ptr->renderScene(world);

	saveToFile("render.bmp", world->width, world->height, world->image);

}

void render_fisheye() {

	std::shared_ptr<World> world{ std::make_shared<World>() };

	create_scene(world);

	world->lights.push_back(
		std::make_shared<Directional>(atlas::math::Vector{ -1, -1, 1 }));
	world->lights[0]->scaleRadiance(1.5f);

	std::shared_ptr<Fisheye> fisheye_ptr(new Fisheye());
	fisheye_ptr->setEye({ 400, -100 , 0 });
	fisheye_ptr->setLookAt({ 100,-32,-700 });
	fisheye_ptr->set_psi_max(50);
	fisheye_ptr->computeUVW();
	fisheye_ptr->renderScene(world);

	saveToFile("fisheye.bmp", world->width, world->height, world->image);
}

void render_directional() {

	std::shared_ptr<World> world{ std::make_shared<World>() };

	create_scene(world);

	world->lights.push_back(
		std::make_shared<Directional>(atlas::math::Vector{ -1, -1, 1 }));
	world->lights[0]->scaleRadiance(1.5f);

	std::shared_ptr<Pinhole> pinhole_ptr(new Pinhole());
	pinhole_ptr->setEye({ 400, -100, 0 });
	pinhole_ptr->setLookAt({ 100,-32,-700 });
	pinhole_ptr->set_view_distance(400);
	pinhole_ptr->computeUVW();
	pinhole_ptr->renderScene(world);

	saveToFile("directional.bmp", world->width, world->height, world->image);
}



int main()
{

	render_pinhole();

	return 0;
}

void saveToFile(std::string const& filename,
	std::size_t width,
	std::size_t height,
	std::vector<Colour> const& image)
{
	std::vector<unsigned char> data(image.size() * 3);

	for (std::size_t i{ 0 }, k{ 0 }; i < image.size(); ++i, k += 3)
	{
		Colour pixel = image[i];
		data[k + 0] = static_cast<unsigned char>(pixel.r * 255);
		data[k + 1] = static_cast<unsigned char>(pixel.g * 255);
		data[k + 2] = static_cast<unsigned char>(pixel.b * 255);
	}

	stbi_write_bmp(filename.c_str(),
		static_cast<int>(width),
		static_cast<int>(height),
		3,
		data.data());
}
