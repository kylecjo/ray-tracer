#pragma once

#define _USE_MATH_DEFINES

#include <atlas/core/Float.hpp>
#include <atlas/math/Math.hpp>
#include <atlas/math/Random.hpp>
#include <atlas/math/Ray.hpp>

#include <fmt/printf.h>
#include <stb_image.h>
#include <stb_image_write.h>

#include <limits>
#include <memory>
#include <vector>

#include <cmath>
#include <iostream>
#include <algorithm>


using atlas::core::areEqual;

using Colour = atlas::math::Vector;
using Ray = atlas::math::Ray<atlas::math::Vector>;

void saveToFile(std::string const& filename,
    std::size_t width,
    std::size_t height,
    std::vector<Colour> const& image);

// Declarations
class BRDF;
class BTDF;
class Camera;
class Material;
class Light;
class Shape;
class Sampler;
class Tracer;
class Texture;
struct ShadeRec;

struct World : std::enable_shared_from_this<World>
{
    std::size_t width, height;
    Colour background;
    std::shared_ptr<Sampler> sampler;
    std::vector<std::shared_ptr<Shape>> scene;
    std::vector<Colour> image;
    std::vector<std::shared_ptr<Light>> lights;
    std::shared_ptr<Light> ambient;
    std::shared_ptr<Tracer> tracer;
    int max_depth;

    Colour max_to_one(const Colour& c) const;
    Colour clamp(const Colour& raw_color) const;
    ShadeRec hit_objects(const Ray& ray);
};

struct ShadeRec
{
    Colour color;
    float t;
    atlas::math::Normal normal;
    atlas::math::Ray<atlas::math::Vector> ray;
    std::shared_ptr<Material> material;
    std::shared_ptr<World> world;
    atlas::math::Point hit_point;
    bool hit_an_object{ false };
    int depth;

};

// Abstract classes defining the interfaces for concrete entities

class Sampler
{
public:
    Sampler(int numSamples, int numSets);
    virtual ~Sampler() = default;

    int getNumSamples() const;

    void setupShuffledIndeces();

    virtual void generateSamples() = 0;

    void map_samples_to_hemisphere(const float e);

    atlas::math::Point sampleUnitSquare();

    atlas::math::Point sampleHemisphere();

protected:
    std::vector<atlas::math::Point> mSamples;
    std::vector<int> mShuffledIndeces;
    std::vector<atlas::math::Point> mHemisphereSamples;

    int mNumSamples;
    int mNumSets;
    unsigned long mCount;
    int mJump;
};

class Shape
{
public:
    Shape();
    virtual ~Shape() = default;

    // if t computed is less than the t in sr, it and the color should be
    // updated in sr
    virtual bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
        ShadeRec& sr) const = 0;

    void setColour(Colour const& col);

    Colour getColour() const;

    void setMaterial(std::shared_ptr<Material> const& material);

    std::shared_ptr<Material> getMaterial() const;

    virtual atlas::math::Point sample() {
        return atlas::math::Point();
    }

    virtual atlas::math::Normal get_normal() {
        return atlas::math::Normal();
    }

    //will be overriden in Rectangle
    virtual atlas::math::Normal get_normal([[maybe_unused]] atlas::math::Point const& p) {
        return atlas::math::Normal();
    }

    virtual float pdf([[maybe_unused]] ShadeRec const& sr) {
        return 1.0f;
    }


    virtual bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
        float& tMin) const = 0;

protected:


    Colour mColour;
    std::shared_ptr<Material> mMaterial;
};

class BRDF
{
public:
    virtual ~BRDF() = default;

    virtual Colour fn(ShadeRec const& sr, atlas::math::Vector const& reflected, atlas::math::Vector const& incoming) const = 0;
    virtual Colour rho(ShadeRec const& sr, atlas::math::Vector const& reflected) const = 0;
};

class BTDF
{
public:
    virtual ~BTDF() = default;

    virtual Colour fn(ShadeRec const& sr, atlas::math::Vector const& reflected, atlas::math::Vector const& incoming) const = 0;
    virtual Colour rho(ShadeRec const& sr, atlas::math::Vector const& reflected) const = 0;
    virtual Colour sample_f(ShadeRec const& sr, atlas::math::Vector const& wo, atlas::math::Vector& wt) const = 0;
    virtual bool tir(ShadeRec const& sr) const = 0;
};

class PerfectTransmitter : public BTDF {
public:
    PerfectTransmitter();

    void set_kt(float k);

    void set_ior(float i);

    Colour fn(ShadeRec const& sr, atlas::math::Vector const& reflected, atlas::math::Vector const& incoming) const;
    Colour rho(ShadeRec const& sr, atlas::math::Vector const& reflected) const;
    Colour sample_f(ShadeRec const& sr, atlas::math::Vector const& wo, atlas::math::Vector& wt) const;
    bool tir(ShadeRec const& sr) const;
private:
    float mKt;
    float mIor;
};

PerfectTransmitter::PerfectTransmitter()
    : BTDF(),
    mKt(NULL),
    mIor(NULL)
{}



class Lambertian : public BRDF
{
public:
    Lambertian();

    Lambertian(float k, Colour c);

    void set_kd(float kd);

    void set_cd(Colour c);

    Colour fn([[maybe_unused]] ShadeRec const& sr,
        [[maybe_unused]] atlas::math::Vector const& reflected,
        [[maybe_unused]] atlas::math::Vector const& incoming) const;

    Colour rho([[maybe_unused]] ShadeRec const& sr,
        [[maybe_unused]] atlas::math::Vector const& reflected) const;

protected:
    Colour mDiffuseColour; //cd
    float mDiffuseReflection; //kd

};

class SV_Lambertian : public BRDF
{
public:
    SV_Lambertian();

    void set_kd(float kd);

    void set_cd(std::shared_ptr<Texture> cd);

    Colour fn([[maybe_unused]] ShadeRec const& sr,
        [[maybe_unused]] atlas::math::Vector const& reflected,
        [[maybe_unused]] atlas::math::Vector const& incoming) const;

    Colour rho([[maybe_unused]] ShadeRec const& sr,
        [[maybe_unused]] atlas::math::Vector const& reflected) const;

private:
    std::shared_ptr<Texture> mCd;
    float mKd; //kd

};

class GlossySpecular : public BRDF
{
public:
    GlossySpecular();

    GlossySpecular(float k, float e, Colour c);

    void set_e(float e);

    void set_ks(float k);

    void set_sampler(std::shared_ptr<Sampler> sampler_ptr);

    void set_cs(Colour c);

    Colour fn([[maybe_unused]] ShadeRec const& sr,
        [[maybe_unused]] atlas::math::Vector const& reflected,
        [[maybe_unused]] atlas::math::Vector const& incoming) const;

    Colour rho([[maybe_unused]] ShadeRec const& sr,
        [[maybe_unused]] atlas::math::Vector const& reflected) const;

    void set_samples(const int num_samples, const float exp);

    Colour sample_f(ShadeRec const&, atlas::math::Vector const& wo, atlas::math::Vector& wi, float& pdf) const;

private:
    float mKs; //ks
    float mPhongExponent; //e
    Colour mCs;
    std::shared_ptr<Sampler> mSamplerPtr;


};

class PerfectSpecular : public BRDF
{
public:
    PerfectSpecular();

    PerfectSpecular(float k, Colour const& c);

    Colour fn([[maybe_unused]] ShadeRec const& sr,
        [[maybe_unused]] atlas::math::Vector const& reflected,
        [[maybe_unused]] atlas::math::Vector const& incoming) const;

    Colour rho([[maybe_unused]] ShadeRec const& sr,
        [[maybe_unused]] atlas::math::Vector const& reflected) const;

    Colour sample_f(ShadeRec const& sr, atlas::math::Vector const& reflected, atlas::math::Vector& incoming) const;

    void set_kr(const float k);

    void set_cr(Colour const& c);
private:
    float mKr;
    Colour mCr;
};

// Materials

class Material
{
public:
    virtual ~Material() = default;

    virtual Colour shade(ShadeRec& sr) = 0;
    virtual Colour area_light_shade([[maybe_unused]] ShadeRec& sr) {
        return Colour{ 0,0,0 };
    }

    //will be overriden
    virtual Colour get_Le([[maybe_unused]] ShadeRec& sr) const {
        return Colour{ 0,0,0 };
    }
};


class Matte : public Material
{
public:
    Matte(std::shared_ptr<Lambertian> ambient_brdf_ptr, std::shared_ptr<Lambertian> diffuse_brdf_ptr);

    void set_cd(Colour c);

    void set_ka(float k);

    void set_kd(float k);


    Colour shade(ShadeRec& sr);
    Colour area_light_shade(ShadeRec& sr);

private:
    std::shared_ptr<Lambertian> mAmbientBRDF;
    std::shared_ptr<Lambertian> mDiffuseBRDF;
};

class SV_Matte : public Material
{
public:
    SV_Matte(std::shared_ptr<SV_Lambertian> ambient_brdf_ptr, std::shared_ptr<SV_Lambertian> diffuse_brdf_ptr);

    void set_cd(std::shared_ptr<Texture> t_ptr);

    void set_ka(float k);

    void set_kd(float k);

    Colour shade(ShadeRec& sr);
    Colour area_light_shade(ShadeRec& sr);

private:
    std::shared_ptr<SV_Lambertian> mAmbientBRDF;
    std::shared_ptr<SV_Lambertian> mDiffuseBRDF;
};

class Phong : public Material {
public:
    Phong(std::shared_ptr<Lambertian> ambient_brdf_ptr, std::shared_ptr<Lambertian> diffuse_brdf_ptr, std::shared_ptr<GlossySpecular> glossy_specular_brdf_ptr);

    Colour shade(ShadeRec& sr);
    Colour area_light_shade(ShadeRec& sr);

    void set_ka(const float k);
    void set_kd(const float k);
    void set_ks(const float k);
    void set_e(const float e);
    void set_cd(Colour c);

private:
    std::shared_ptr<Lambertian> mAmbientBRDF;
    std::shared_ptr<Lambertian> mDiffuseBRDF;
    std::shared_ptr<GlossySpecular> mSpecularBRDF;
};

class GlossyReflector : public Phong {
public:
    GlossyReflector(std::shared_ptr<Lambertian> ambient_brdf_ptr,
        std::shared_ptr<Lambertian> diffuse_brdf_ptr,
        std::shared_ptr<GlossySpecular> glossy_specular_brdf_ptr,
        std::shared_ptr<GlossySpecular> glossy_specular_brdf_ptr2);

    void set_samples(const int num_samples, const float exp);
    void set_kr(const float k);
    void set_exponent(const float exp);
    void set_cr(Colour c);

    Colour shade(ShadeRec& sr);

private:
    std::shared_ptr<GlossySpecular> mGlossySpecularBRDF;
};

class Reflective : public Phong {
public:
    Reflective(std::shared_ptr<Lambertian> ambient_brdf_ptr, std::shared_ptr<Lambertian> diffuse_brdf_ptr, std::shared_ptr<GlossySpecular> glossy_specular_brdf_ptr, std::shared_ptr<PerfectSpecular> perfect_specular_brdf_ptr);

    Colour shade(ShadeRec& sr);

    void set_kr(const float k);

    void set_cr(Colour const& c);

private:

    std::shared_ptr<PerfectSpecular> mPerfectSpecularBRDF;
};

class Emissive : public Material {
public:
    Emissive();

    Emissive(float l, Colour c);

    void scale_radiance(float ls);

    void set_ce(Colour c);

    Colour get_Le([[maybe_unused]] ShadeRec& sr) const;

    Colour shade([[maybe_unused]] ShadeRec& sr);

    Colour area_light_shade([[maybe_unused]] ShadeRec& sr);

private:
    float mLs;
    Colour mCe;
};

class Transparent : public Phong {
public:
    Transparent(std::shared_ptr<Lambertian> ambient_brdf_ptr, std::shared_ptr<Lambertian> diffuse_brdf_ptr,
        std::shared_ptr<GlossySpecular> glossy_specular_brdf_ptr, std::shared_ptr<PerfectSpecular> perfect_specular_brdf_ptr,
        std::shared_ptr<PerfectTransmitter> perfect_transmitter_btdf_ptr);

    Colour shade(ShadeRec& sr);

    void set_kt(float k);

    void set_ior(float i);

    void set_kr(float k);

    void set_cr(Colour c);

private:
    std::shared_ptr<PerfectSpecular> mPerfectSpecularBRDF; //reflective BRDF
    std::shared_ptr<PerfectTransmitter> mPerfectTransmitterBTDF; //specular BTDF
};





// Lights

class Light
{
public:

    Light() {
        mColour = { 1,1,1 };
        mRadiance = 0.2f;
        mShadows = true;
    }

    virtual atlas::math::Vector getDirection(ShadeRec& sr) = 0;
    // this is omega i in the rendering equation

    virtual Colour L(ShadeRec& sr);
    //this is ls*cl in the rendering equation

    bool castsShadows() const;
    void scaleRadiance(float b);
    void setColour(Colour const& c);

    virtual float G([[maybe_unused]] ShadeRec const& sr) const {
        return 1.0f;
    }

    virtual float pdf([[maybe_unused]] ShadeRec const& sr) const {
        return 1.0f;
    }

    virtual bool in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const = 0;

protected:
    Colour mColour;
    float mRadiance;
    bool mShadows;
};


class Ambient : public Light {

public:
    Ambient()
    {
        mColour = { 1, 1, 1 };
        mRadiance = 0.2f;
    }

    atlas::math::Vector getDirection([[maybe_unused]] ShadeRec& sr) {
        return { 0,0,0 }; // returns 0 because not called anywhere
    }

    Colour L([[maybe_unused]] ShadeRec& sr) {
        return mRadiance * mColour;
    }

    bool in_shadow([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray, [[maybe_unused]] ShadeRec& sr) const {
        return(false);
    }

private:
    Colour mColour;
    float mRadiance;


};

class AmbientOccluder : public Light {
public:
    AmbientOccluder();

    AmbientOccluder(Colour min);

    void set_sampler(std::shared_ptr<Sampler> sampler_ptr);

    void set_min_amount(Colour min);

    atlas::math::Vector getDirection(ShadeRec& sr);

    bool in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const;

    Colour L(ShadeRec& sr);

private:
    atlas::math::Vector mU, mV, mW;
    std::shared_ptr<Sampler> mSamplerPtr;
    Colour mMinAmount;

};

class Directional : public Light {
public:
    Directional(atlas::math::Vector dir) {
        mColour = { 1,1,1 };
        mRadiance = 1.0f;
        mDir = dir;
    }

    atlas::math::Vector getDirection([[maybe_unused]] ShadeRec& sr) {
        return mDir;
        // get the normal from the shaderec and dot product with vector going towards light
    }

    Colour L([[maybe_unused]] ShadeRec& sr) {
        return mRadiance * mColour;
    }

    bool in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const;


private:
    Colour mColour;
    float mRadiance;
    atlas::math::Vector mDir;
};

class PointL : public Light {
public:
    PointL(atlas::math::Point loc) {
        mColour = { 1,1,1 };
        mRadiance = 1.0f;
        mLocation = loc;
    }

    atlas::math::Vector getDirection([[maybe_unused]] ShadeRec& sr) {
        return glm::normalize(mLocation - sr.hit_point);
        // get the normal from the shaderec and dot product with vector going towards light
    }

    Colour L([[maybe_unused]] ShadeRec& sr) {
        return mRadiance * mColour;
    }

    bool in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const;

private:
    Colour mColour;
    float mRadiance;
    atlas::math::Point mLocation;
};

class AreaLight : public Light {
public:
    //constructors
    AreaLight(void)
        : Light()
    {}

    //access functions

    void set_object(std::shared_ptr<Shape> shape_ptr) {
        mShapePtr = shape_ptr;
        if (mShapePtr) {
            mMaterialPtr = shape_ptr->getMaterial();
        }
    }

    atlas::math::Vector getDirection(ShadeRec& sr) {
        mSamplePoint = mShapePtr->sample();
        mLightNormal = mShapePtr->get_normal(mSamplePoint);
        mWi = mSamplePoint - sr.hit_point;
        mWi = glm::normalize(mWi);

        return mWi;
    }
    bool in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const {
        float t;
        float ts = glm::dot((mSamplePoint - ray.o), ray.d);

        for (int j = 0; j < sr.world->scene.size(); j++) {
            if (sr.world->scene[j]->intersectRay(ray, t) && t < ts)
                return true;
        }
        return false;
    }
    Colour L(ShadeRec& sr) {
        float ndotd = glm::dot(-mLightNormal, mWi); //weird stuff going on with the normal here, it should be negative but flipped the sign to make it work

        if (ndotd > 0.0f) {
            return mMaterialPtr->get_Le(sr);
        }
        else
            return { 0, 0, 0 };
    }
    float G(ShadeRec const& sr) const {
        float ndotd = glm::dot(-mLightNormal, mWi);
        float d2 = glm::distance2(mSamplePoint, sr.hit_point);

        return (ndotd / d2);

    }
    float pdf(ShadeRec const& sr) const {
        return mShapePtr->pdf(sr);
    }



private:
    std::shared_ptr<Shape> mShapePtr;
    std::shared_ptr<Material> mMaterialPtr;
    atlas::math::Point mSamplePoint;
    atlas::math::Normal mLightNormal;
    atlas::math::Vector mWi;
};

// Concrete classes which we can construct and use in our ray tracer

class Sphere : public Shape
{
public:
    Sphere(atlas::math::Point center, float radius);

    bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
        ShadeRec& sr) const;



private:
    bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
        float& tMin) const;

    atlas::math::Point mCentre;
    float mRadius;
    float mRadiusSqr;
};

class Triangle : public Shape
{
public:
    Triangle(atlas::math::Point p0, atlas::math::Point p1, atlas::math::Point p2, atlas::math::Vector n);

    bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
        ShadeRec& sr) const;

private:
    bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
        float& tMin) const;

    atlas::math::Point mP0;
    atlas::math::Point mP1;
    atlas::math::Point mP2;
    atlas::math::Vector mNorm;


};

class Rectangle : public Shape
{
public:
    Rectangle(atlas::math::Point p0, atlas::math::Vector a, atlas::math::Vector b);

    bool hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const;

    void set_sampler(std::shared_ptr<Sampler> sampler_ptr) {
        mSamplerPtr = sampler_ptr;
    }

    atlas::math::Point sample() {
        atlas::math::Point2 sample_point = mSamplerPtr->sampleUnitSquare();
        return (mP0 + sample_point.x * mA + sample_point.y * mB);
    }

    float pdf([[maybe_unused]] ShadeRec& sr) {
        return mInvArea;
    }

    atlas::math::Normal get_normal() {
        return mNormal;
    }

    atlas::math::Normal get_normal([[maybe_unused]] atlas::math::Point const& p) {
        return mNormal;
    }

private:
    bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const;

    atlas::math::Point mP0;
    atlas::math::Vector mA;
    atlas::math::Vector mB;
    atlas::math::Normal mNormal;
    std::shared_ptr<Sampler> mSamplerPtr;
    float mInvArea;
};

class Plane : public Shape
{
public:
    Plane(atlas::math::Point po, atlas::math::Vector norm);

    bool hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const;

private:
    bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const;

    atlas::math::Point mPoint;
    atlas::math::Vector mNormal;

};

class Box : public Shape
{
public:
    Box(atlas::math::Point p1, atlas::math::Point p2);

    bool hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const;

private:
    bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tMin) const;
    atlas::math::Vector get_normal(const int face_hit) const;

    atlas::math::Point mPoint1;
    atlas::math::Point mPoint2;

};

// MultiSampling

class Regular : public Sampler
{
public:
    Regular(int numSamples, int numSets);

    void generateSamples();
};

class Random : public Sampler
{
public:
    Random(int numSamples, int numSets);

    void generateSamples();
};

class Jitter : public Sampler
{
public:
    Jitter(int numSamples, int numSets);

    void generateSamples();
};




// Cameras

class Camera
{
public:
    Camera();

    virtual ~Camera() = default;

    virtual void renderScene(std::shared_ptr<World> world) const = 0;

    void setEye(atlas::math::Point const& eye);

    void setLookAt(atlas::math::Point const& lookAt);

    void setUpVector(atlas::math::Vector const& up);

    void computeUVW();

protected:
    atlas::math::Point mEye;
    atlas::math::Point mLookAt;
    atlas::math::Point mUp;
    atlas::math::Vector mU, mV, mW;
};

class Pinhole : public Camera {
public:

    Pinhole(void)
        : Camera(),
        d{ 0.0f }, zoom{ 0.0f }
    {}

    void renderScene(std::shared_ptr<World> world) const;

    atlas::math::Vector ray_direction(const atlas::math::Point2& p) const;

    void set_view_distance(float vd)
    {
        d = vd;
    }

private:
    float d;
    float zoom;
};

class Fisheye : public Camera {
public:

    Fisheye(void)
        : Camera(),
        mPsiMax{ 0.0f }
    {}

    void renderScene(std::shared_ptr<World> world) const;

    atlas::math::Vector ray_direction(const atlas::math::Point2& p, const size_t width,
        const size_t height, float& r_squared) const;

    void set_psi_max(float psi) {
        mPsiMax = psi;
    }

private:
    float mPsiMax;
};

class Tracer {
public:
    Tracer(std::shared_ptr<World> world_ptr);

    //virtual Colour trace_ray(ShadeRec const& sr, const Ray& ray) const;

    virtual Colour trace_ray(const Ray& ray) const = 0;

    virtual Colour trace_ray(const Ray ray, const int depth) const = 0;

protected:

    std::shared_ptr<World> mWorld;
};

class RayCast : public Tracer {
public:

    RayCast(std::shared_ptr<World> world_ptr);

    Colour trace_ray(const Ray& ray) const;
    Colour trace_ray(const Ray ray, const int depth) const;
};

class Whitted : public Tracer {
public:

    Whitted(std::shared_ptr<World> world_ptr);

    Colour trace_ray(const Ray& ray) const;
    Colour trace_ray(const Ray ray, const int depth) const;
};

class AreaLighting : public Tracer {
public:

    AreaLighting(std::shared_ptr<World> world_ptr);

    Colour trace_ray(const Ray& ray) const;
    Colour trace_ray(const Ray ray, const int depth) const;
};

class Texture {
public:
    Texture();
    virtual ~Texture() = default;

    virtual Colour get_colour(ShadeRec const& sr) const = 0;
};

class Checker3D : public Texture {
public:
    Checker3D();

    Checker3D(Colour c1, Colour c2, float s);

    void set_size(float s);

    void set_colour1(Colour c1);

    void set_colour2(Colour c2);

    Colour get_colour(ShadeRec const& sr) const;

private:
    Colour mColour1;
    Colour mColour2;
    float mSize;
};



