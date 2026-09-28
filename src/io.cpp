#include "blitz/io.hpp"
#define CGLTF_IMPLEMENTATION
#include "cgltf.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <bit>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <memory>
#include <stdexcept>
namespace blitz {
namespace fs=std::filesystem;
using json=nlohmann::json;
namespace {
[[noreturn]] void fail(const std::string& s){throw std::runtime_error(s);}
template<class T> T little(std::istream& f) {
    std::array<std::byte,sizeof(T)> b{};f.read(reinterpret_cast<char*>(b.data()),b.size());
    if(!f)fail("truncated binary mesh");
    if constexpr(std::endian::native==std::endian::big)std::reverse(b.begin(),b.end());
    return std::bit_cast<T>(b);
}
template<class T> void append(std::vector<uint8_t>& bytes,T v) {
    auto b=std::bit_cast<std::array<uint8_t,sizeof(T)>>(v);
    if constexpr(std::endian::native==std::endian::big)std::reverse(b.begin(),b.end());
    bytes.insert(bytes.end(),b.begin(),b.end());
}
Mesh gltf(const fs::path& path,std::optional<size_t> selected={}) {
    cgltf_options opts{};cgltf_data* raw{};
    auto filename=path.string();
    if(cgltf_parse_file(&opts,filename.c_str(),&raw)!=cgltf_result_success)fail("cannot parse glTF");
    std::unique_ptr<cgltf_data,decltype(&cgltf_free)> doc(raw,cgltf_free);auto& d=*doc;
    if(d.skins_count||d.animations_count)fail("skinning and animation are outside the static-mesh contract");
    if(d.materials_count>=UINT16_MAX)fail("too many materials");
    if(cgltf_load_buffers(&opts,&d,filename.c_str())!=cgltf_result_success||cgltf_validate(&d)!=cgltf_result_success)fail("invalid glTF buffers or accessors");
    for(size_t i=0;i<d.extensions_required_count;++i) {
        std::string ext=d.extensions_required[i];
        if(ext!="KHR_mesh_quantization"&&ext!="KHR_materials_unlit"&&ext!="KHR_texture_transform")fail("unsupported required glTF extension: "+ext);
    }
    Mesh out;out.double_sided.resize(d.materials_count+1);
    for(size_t i=0;i<d.materials_count;++i) {
        if(d.materials[i].alpha_mode!=cgltf_alpha_mode_opaque||d.materials[i].has_transmission||d.materials[i].has_diffuse_transmission)fail("nonopaque glTF material");
        out.double_sided[i]=d.materials[i].double_sided;
    }
    bool any_normal=false,any_uv=false,any_color=false,any_tangent=false;
    auto add=[&](cgltf_mesh* mesh,const float* t) {
        double det=double(t[0])*(double(t[5])*t[10]-double(t[9])*t[6])-double(t[4])*(double(t[1])*t[10]-double(t[9])*t[2])+double(t[8])*(double(t[1])*t[6]-double(t[5])*t[2]);
        if(!std::isfinite(det)||std::abs(det)<1e-30)fail("singular node transform");
        auto direction=[&](Vec3 n,bool normal) {
            if(!normal)return normalized({t[0]*n.x+t[4]*n.y+t[8]*n.z,t[1]*n.x+t[5]*n.y+t[9]*n.z,t[2]*n.x+t[6]*n.y+t[10]*n.z});
            Vec3 a{t[0],t[1],t[2]},b{t[4],t[5],t[6]},c{t[8],t[9],t[10]};
            return normalized((cross(b,c)*n.x+cross(c,a)*n.y+cross(a,b)*n.z)*(1/det));
        };
        for(size_t k=0;k<mesh->primitives_count;++k) {
            auto& p=mesh->primitives[k];
            if(p.type!=cgltf_primitive_type_triangles||p.targets_count||p.has_draco_mesh_compression)fail("unsupported primitive (triangles without morphs or compression required)");
            cgltf_accessor *pos{},*norm{},*uv{},*col{},*tan{};
            for(size_t a=0;a<p.attributes_count;++a) {
                auto& at=p.attributes[a];
                if(at.type==cgltf_attribute_type_position)pos=at.data;
                if(at.type==cgltf_attribute_type_normal)norm=at.data;
                if(at.type==cgltf_attribute_type_texcoord&&at.index==0)uv=at.data;
                if(at.type==cgltf_attribute_type_color&&at.index==0)col=at.data;
                if(at.type==cgltf_attribute_type_tangent)tan=at.data;
            }
            if(!pos||pos->type!=cgltf_type_vec3||pos->count+out.positions.size()>=UINT32_MAX)fail("missing or oversized POSITION");
            auto unpack=[&](cgltf_accessor* a,size_t components) {
                if(!a)return std::vector<float>{};
                if(a->count!=pos->count||cgltf_num_components(a->type)!=components)fail("attribute shape mismatch");
                std::vector<float> v(a->count*components);
                if(cgltf_accessor_unpack_floats(a,v.data(),v.size())!=v.size())fail("cannot unpack attribute");
                return v;
            };
            auto pv=unpack(pos,3),nv=unpack(norm,3),uvv=unpack(uv,2),tv=unpack(tan,4);
            size_t cc=col?cgltf_num_components(col->type):4;
            if(cc!=3&&cc!=4)fail("invalid vertex color shape");
            auto cv=unpack(col,cc);
            uint32_t base=uint32_t(out.positions.size());
            for(size_t i=0;i<pos->count;++i) {
                auto x=pv[i*3],y=pv[i*3+1],z=pv[i*3+2];
                out.positions.push_back({t[0]*x+t[4]*y+t[8]*z+t[12],t[1]*x+t[5]*y+t[9]*z+t[13],t[2]*x+t[6]*y+t[10]*z+t[14]});
                out.normals.push_back(norm?direction({nv[i*3],nv[i*3+1],nv[i*3+2]},true):Vec3{});
                out.uv.push_back(uv?Vec2{uvv[i*2],uvv[i*2+1]}:Vec2{});
                out.colors.push_back(col?Vec4{cv[i*cc],cv[i*cc+1],cv[i*cc+2],cc==4?cv[i*cc+3]:1}:Vec4{1,1,1,1});
                auto v=tan?direction({tv[i*4],tv[i*4+1],tv[i*4+2]},false):Vec3{};
                out.tangents.push_back({v.x,v.y,v.z,tan?float(tv[i*4+3]*(det<0?-1:1)):1});
            }
            any_normal|=bool(norm);any_uv|=bool(uv);any_color|=bool(col);any_tangent|=bool(tan);
            size_t count=p.indices?p.indices->count:pos->count;
            if(count%3||count+out.indices.size()>=UINT32_MAX)fail("invalid triangle index count");
            for(size_t i=0;i<count;i+=3) {
                uint32_t tri[3];
                for(int j=0;j<3;++j){size_t index=p.indices?cgltf_accessor_read_index(p.indices,i+j):i+j;if(index>=pos->count)fail("index outside primitive");tri[j]=base+uint32_t(index);}
                if(det<0)std::swap(tri[1],tri[2]);
                out.indices.insert(out.indices.end(),tri,tri+3);
                out.materials.push_back(p.material?uint16_t(p.material-d.materials):uint16_t(d.materials_count));
            }
        }
    };
    std::vector<uint8_t> visited(d.nodes_count);
    std::function<void(cgltf_node*,unsigned)> node=[&](cgltf_node* n,unsigned depth) {
        size_t id=size_t(n-d.nodes);if(id>=visited.size()||visited[id]||depth>256)fail("cyclic or multiply parented scene");
        visited[id]=1;if(n->has_mesh_gpu_instancing)fail("GPU instancing extension is unsupported");
        if(n->mesh){float t[16];cgltf_node_transform_world(n,t);add(n->mesh,t);}
        for(size_t i=0;i<n->children_count;++i)node(n->children[i],depth+1);
    };
    if(selected) {
        if(*selected>=d.meshes_count)fail("glTF mesh index out of range");
        float t[16]={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};add(d.meshes+*selected,t);
    }
    else if(d.scene){for(size_t i=0;i<d.scene->nodes_count;++i)node(d.scene->nodes[i],0);}
    else if(d.nodes_count){for(size_t i=0;i<d.nodes_count;++i)if(!d.nodes[i].parent)node(d.nodes+i,0);}
    else {float t[16]={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};for(size_t i=0;i<d.meshes_count;++i)add(d.meshes+i,t);}
    if(!any_normal)out.normals.clear();if(!any_uv)out.uv.clear();if(!any_color)out.colors.clear();if(!any_tangent)out.tangents.clear();
    return out;
}
Mesh obj(const fs::path& path) {
    tinyobj::ObjReader reader;tinyobj::ObjReaderConfig config;config.triangulate=true;
    config.mtl_search_path=path.parent_path().string();
    if(!reader.ParseFromFile(path.string(),config))fail("OBJ: "+reader.Error());
    Mesh out;auto& a=reader.GetAttrib();bool normals=false,uv=false;
    struct Key {int v,n,t;bool operator==(const Key&)const=default;};
    struct Hash {size_t operator()(Key k)const{return size_t(uint32_t(k.v))*73856093u^size_t(uint32_t(k.n))*19349663u^size_t(uint32_t(k.t))*83492791u;}};
    std::unordered_map<Key,uint32_t,Hash> vertices;
    if(reader.GetMaterials().size()>=UINT16_MAX)fail("too many OBJ materials");
    for(auto& material:reader.GetMaterials())if(material.dissolve<1||!material.alpha_texname.empty())fail("nonopaque OBJ material");
    for(auto& shape:reader.GetShapes()) {
        size_t offset=0;
        for(size_t f=0;f<shape.mesh.num_face_vertices.size();++f) {
            if(shape.mesh.num_face_vertices[f]!=3)fail("OBJ triangulation failed");
            for(int k=0;k<3;++k) {
                auto index=shape.mesh.indices[offset++];
                Key key{index.vertex_index,index.normal_index,index.texcoord_index};
                auto [it,inserted]=vertices.emplace(key,uint32_t(out.positions.size()));
                if(inserted) {
                    if(key.v<0||size_t(key.v)*3+2>=a.vertices.size())fail("invalid OBJ position index");
                    out.positions.push_back({a.vertices[key.v*3],a.vertices[key.v*3+1],a.vertices[key.v*3+2]});
                    if(key.n>=0&&size_t(key.n)*3+2>=a.normals.size())fail("invalid OBJ normal index");
                    if(key.t>=0&&size_t(key.t)*2+1>=a.texcoords.size())fail("invalid OBJ UV index");
                    out.normals.push_back(key.n<0?Vec3{}:Vec3{a.normals[key.n*3],a.normals[key.n*3+1],a.normals[key.n*3+2]});
                    out.uv.push_back(key.t<0?Vec2{}:Vec2{a.texcoords[key.t*2],a.texcoords[key.t*2+1]});
                    normals|=key.n>=0;uv|=key.t>=0;
                    if(a.colors.size()==a.vertices.size())out.colors.push_back({a.colors[key.v*3],a.colors[key.v*3+1],a.colors[key.v*3+2],1});
                }
                out.indices.push_back(it->second);
            }
            out.materials.push_back(uint16_t(shape.mesh.material_ids[f]+1));
        }
    }
    if(!normals)out.normals.clear();if(!uv)out.uv.clear();return out;
}
Mesh ply(const fs::path& path) {
    std::ifstream f(path,std::ios::binary);std::string line;
    std::getline(f,line);if(line!="ply"&&line!="ply\r")fail("invalid PLY magic");
    struct Prop {std::string name,type,count;};struct Element {std::string name;size_t count;std::vector<Prop> props;};
    std::vector<Element> elements;bool ascii=true,big=false,ended=false;
    while(std::getline(f,line)) {
        std::istringstream s(line);std::string word;s>>word;
        if(word=="end_header"){ended=true;break;}
        if(word=="format"){s>>word;if(word=="binary_little_endian")ascii=false;else if(word=="binary_big_endian"){ascii=false;big=true;}else if(word!="ascii")fail("unknown PLY format");}
        if(word=="element"){Element e;s>>e.name>>e.count;if(!s||e.count>=UINT32_MAX)fail("oversized PLY element");elements.push_back(e);}
        if(word=="property"){if(elements.empty())fail("PLY property without element");Prop p;s>>p.type;if(p.type=="list"){s>>p.count>>p.type;}s>>p.name;elements.back().props.push_back(p);}
    }
    if(!ended)fail("missing PLY end_header");
    auto value=[&](const std::string& type)->double {
        if(ascii){double v;if(!(f>>v)||!std::isfinite(v))fail("bad PLY value");return v;}
        auto read=[&]<class T>()->double {
            auto v=little<T>(f);if(big){auto b=std::bit_cast<std::array<std::byte,sizeof(T)>>(v);std::reverse(b.begin(),b.end());v=std::bit_cast<T>(b);}return double(v);
        };
        if(type=="float"||type=="float32")return read.operator()<float>();if(type=="double"||type=="float64")return read.operator()<double>();
        if(type=="uchar"||type=="uint8")return read.operator()<uint8_t>();if(type=="char"||type=="int8")return read.operator()<int8_t>();
        if(type=="ushort"||type=="uint16")return read.operator()<uint16_t>();if(type=="short"||type=="int16")return read.operator()<int16_t>();
        if(type=="uint"||type=="uint32")return read.operator()<uint32_t>();if(type=="int"||type=="int32")return read.operator()<int32_t>();
        fail("unknown PLY scalar type");
    };
    Mesh m;bool normals=false,uv=false,colors=false;
    for(auto& e:elements)for(size_t i=0;i<e.count;++i) {
        Vec3 p{},n{};Vec2 t{};Vec4 c{1,1,1,1};uint16_t mat=0;
        for(auto& prop:e.props) {
            if(!prop.count.empty()) {
                double raw=value(prop.count);if(raw<0||raw>1000000||std::floor(raw)!=raw)fail("bad PLY list length");
                size_t count=size_t(raw);bool indices=e.name=="face"&&(prop.name=="vertex_indices"||prop.name=="vertex_index");
                if(indices&&count!=3)fail("PLY requires triangulated faces");
                for(size_t k=0;k<count;++k){double v=value(prop.type);if(indices){if(v<0||v>=UINT32_MAX||std::floor(v)!=v)fail("bad PLY index");m.indices.push_back(uint32_t(v));}}
            } else {
                double v=value(prop.type);
                if(e.name=="vertex") {
                    if(prop.name=="x")p.x=float(v);if(prop.name=="y")p.y=float(v);if(prop.name=="z")p.z=float(v);
                    if(prop.name=="nx"){n.x=float(v);normals=true;}if(prop.name=="ny")n.y=float(v);if(prop.name=="nz")n.z=float(v);
                    if(prop.name=="u"||prop.name=="s"){t.x=float(v);uv=true;}if(prop.name=="v"||prop.name=="t")t.y=float(v);
                    double cv=(prop.type=="uchar"||prop.type=="uint8")?v/255:v;
                    if(prop.name=="red"){c.x=float(cv);colors=true;}if(prop.name=="green")c.y=float(cv);if(prop.name=="blue")c.z=float(cv);
                } else if(e.name=="face"&&prop.name=="material_index"){if(v<0||v>65535||std::floor(v)!=v)fail("invalid material index");mat=uint16_t(v);}
            }
        }
        if(e.name=="vertex"){m.positions.push_back(p);m.normals.push_back(n);m.uv.push_back(t);m.colors.push_back(c);}
        if(e.name=="face")m.materials.push_back(mat);
    }
    if(!normals)m.normals.clear();if(!uv)m.uv.clear();if(!colors)m.colors.clear();return m;
}
Mesh stl(const fs::path& path) {
    std::ifstream f(path,std::ios::binary);auto bytes=fs::file_size(path);uint32_t count=0;
    if(bytes>=84){f.seekg(80);count=little<uint32_t>(f);}bool binary=bytes>=84&&bytes==84ull+50ull*count;
    Mesh m;std::unordered_map<std::string,uint32_t> vertices;
    auto add=[&](Vec3 v){auto key=std::string(reinterpret_cast<const char*>(&v),sizeof(v));auto [it,ok]=vertices.emplace(key,uint32_t(m.positions.size()));if(ok)m.positions.push_back(v);m.indices.push_back(it->second);};
    if(binary)for(uint32_t i=0;i<count;++i){for(int j=0;j<3;++j)(void)little<float>(f);for(int j=0;j<3;++j){Vec3 p;p.x=little<float>(f);p.y=little<float>(f);p.z=little<float>(f);add(p);}(void)little<uint16_t>(f);}
    else {f.clear();f.seekg(0);std::string word;while(f>>word)if(word=="vertex"){Vec3 p;if(!(f>>p.x>>p.y>>p.z))fail("invalid ASCII STL");add(p);}}
    return m;
}
json measurement(const Measurement& m) {
    return {{"error_px",m.error},{"coverage_px",m.coverage},{"coverage_upper_px",m.coverage_upper},{"changed_area",m.changed_area},
      {"normal_degrees",m.normal_degrees},{"worst_view",m.worst_view},{"views_evaluated",m.views_evaluated},
      {"supersample",m.supersample},{"complete",m.complete},{"passed",m.passed},{"resource_limited",m.resource_limited},{"nonfinite_error",!std::isfinite(m.error)}};
}
}
Mesh load_mesh(const fs::path& p) {
    if(fs::file_size(p)>1024ull*1024*1024)fail("input exceeds 1 GiB import cap");
    auto ext=p.extension().string();std::transform(ext.begin(),ext.end(),ext.begin(),[](unsigned char c){return char(std::tolower(c));});
    Mesh m;if(ext==".gltf"||ext==".glb")m=gltf(p);else if(ext==".obj")m=obj(p);else if(ext==".ply")m=ply(p);else if(ext==".stl")m=stl(p);else fail("unsupported mesh extension");
    if(auto e=validate(m.view());!e.empty())fail(p.string()+": "+e);
    return m;
}
Mesh load_gltf_mesh(const fs::path& p,size_t index) {
    if(fs::file_size(p)>1024ull*1024*1024)fail("input exceeds 1 GiB import cap");
    auto m=gltf(p,index);
    if(auto e=validate(m.view());!e.empty())fail(p.string()+": "+e);
    return m;
}
void save_ply(MeshView m,const fs::path& path) {
    std::ofstream f(path,std::ios::binary);f<<"ply\nformat binary_little_endian 1.0\nelement vertex "<<m.positions.count<<"\nproperty float x\nproperty float y\nproperty float z\n";
    if(m.normals)f<<"property float nx\nproperty float ny\nproperty float nz\n";
    if(m.uv)f<<"property float u\nproperty float v\n";
    if(m.colors)f<<"property float red\nproperty float green\nproperty float blue\n";
    f<<"element face "<<m.triangles()<<"\nproperty list uchar uint vertex_indices\nproperty ushort material_index\nend_header\n";
    std::vector<uint8_t> row;auto flush=[&]{f.write(reinterpret_cast<char*>(row.data()),row.size());row.clear();};
    for(size_t i=0;i<m.positions.count;++i){auto p=m.positions[i];append(row,p.x);append(row,p.y);append(row,p.z);
        if(m.normals){auto n=m.normals[i];append(row,n.x);append(row,n.y);append(row,n.z);}
        if(m.uv){auto u=m.uv[i];append(row,u.x);append(row,u.y);}
        if(m.colors){auto c=m.colors[i];append(row,c.x);append(row,c.y);append(row,c.z);}flush();}
    for(size_t f0=0;f0<m.triangles();++f0){append(row,uint8_t(3));for(int k=0;k<3;++k)append(row,m.indices[f0*3+k]);append(row,m.material(f0));flush();}
    if(!f)fail("cannot write PLY");
}
json result_json(const Result& r) {
    json j={{"version",1},{"status",r.status==Status::Complete?"complete":r.status==Status::Cancelled?"cancelled":"budget_limited"},
      {"candidate_evaluations",r.candidate_evaluations},{"lods",json::array()}};
    auto runtime=runtime_levels(r);j["runtime_levels"]=runtime;j["runtime_lod_count"]=runtime.size();
    const char* stages[]={"source_search","adjacent_search","source_audit","adjacent_audit"};
    for(size_t i=0;i<4;++i)j["rejections"][stages[i]]={{"count",r.rejected_gates[i]},{"worst",r.rejected_gates[i]?measurement(r.worst_rejected[i]):json(nullptr)}};
    for(auto& l:r.lods){auto v=l.view(r.source);auto d=uv_distortion(v);j["lods"].push_back({
      {"triangles",v.triangles()},{"vertices",v.positions.count},{"shared_vertices",l.shared_vertices},
      {"screen_pixels",l.schedule.pixels},{"transition_limit",l.schedule.transition},{"source_limit",l.schedule.source},
      {"adjacent",measurement(l.adjacent)},{"source",measurement(l.source_error)},
      {"uv_diagnostics",{{"mean_density",d.mean_uv_density},{"max_density",d.max_uv_density},{"max_anisotropy",d.max_uv_anisotropy},
          {"negative_winding_faces",d.negative_uv_faces},{"degenerate_uv_faces",d.degenerate_uv_faces}}}});}
    return j;
}
void save_chain(const Result& r,const fs::path& directory) {
    fs::create_directories(directory);json j={{"asset",{{"version","2.0"},{"generator","BlitzRemesher"}}},{"scene",0},{"scenes",json::array({{{"nodes",json::array({0})}}})},
      {"nodes",json::array()},{"meshes",json::array()},{"accessors",json::array()},{"bufferViews",json::array()},{"materials",json::array()}};
    uint16_t maxmat=0;for(auto& l:r.lods)for(size_t f=0;f<l.data.indices.size()/3;++f)maxmat=std::max(maxmat,l.view(r.source).material(f));
    for(unsigned i=0;i<=maxmat;++i)j["materials"].push_back({{"name","material_"+std::to_string(i)},{"doubleSided",i<r.source.double_sided.size()&&r.source.double_sided[i]!=0}});
    std::vector<uint8_t> bytes;
    auto accessor=[&](size_t offset,size_t count,int components,int type,const char* shape) {
        size_t view=j["bufferViews"].size(),id=j["accessors"].size();
        j["bufferViews"].push_back({{"buffer",0},{"byteOffset",offset},{"byteLength",count*components*4}});
        j["accessors"].push_back({{"bufferView",view},{"componentType",type},{"count",count},{"type",shape}});return id;
    };
    auto attributes=[&](MeshView v) {
        json a;
        auto vec=[&](auto stream,const char* name,int n,const char* shape) {
            if(!stream)return;size_t start=bytes.size();
            for(size_t i=0;i<stream.count;++i){auto p=stream[i];const float* fields=reinterpret_cast<const float*>(&p);for(int k=0;k<n;++k)append(bytes,fields[k]);}
            a[name]=accessor(start,stream.count,n,5126,shape);
        };
        vec(v.positions,"POSITION",3,"VEC3");vec(v.normals,"NORMAL",3,"VEC3");vec(v.uv,"TEXCOORD_0",2,"VEC2");vec(v.colors,"COLOR_0",4,"VEC4");vec(v.tangents,"TANGENT",4,"VEC4");
        Vec3 lo=v.positions[0],hi=lo;for(size_t i=1;i<v.positions.count;++i){auto p=v.positions[i];lo={std::min(lo.x,p.x),std::min(lo.y,p.y),std::min(lo.z,p.z)};hi={std::max(hi.x,p.x),std::max(hi.y,p.y),std::max(hi.z,p.z)};}
        j["accessors"][a["POSITION"].get<size_t>()]["min"]={lo.x,lo.y,lo.z};j["accessors"][a["POSITION"].get<size_t>()]["max"]={hi.x,hi.y,hi.z};return a;
    };
    json shared=attributes(r.source);
    const auto runtime=runtime_levels(r);
    for(auto i:runtime) {
        auto v=r.lods[i].view(r.source);auto a=r.lods[i].shared_vertices?shared:attributes(v);json primitives=json::array();
        std::map<uint16_t,std::vector<uint32_t>> groups;
        for(size_t f=0;f<v.triangles();++f)for(int k=0;k<3;++k)groups[v.material(f)].push_back(v.indices[f*3+k]);
        for(auto& [mat,indices]:groups){size_t offset=bytes.size();for(auto index:indices)append(bytes,index);
            primitives.push_back({{"attributes",a},{"indices",accessor(offset,indices.size(),1,5125,"SCALAR")},{"material",mat},{"mode",4}});}
        j["meshes"].push_back({{"name","LOD"+std::to_string(i)},{"primitives",primitives}});
    }
    size_t mesh_index=0;
    for(size_t i=0;i<r.lods.size();++i) {
        if(mesh_index+1<runtime.size()&&i==runtime[mesh_index+1])++mesh_index;
        j["nodes"].push_back({{"mesh",mesh_index},{"name","LOD"+std::to_string(i)}});
    }
    j["buffers"]=json::array({{{"uri","chain.bin"},{"byteLength",bytes.size()}}});
    std::ofstream bin(directory/"chain.bin",std::ios::binary);bin.write(reinterpret_cast<char*>(bytes.data()),bytes.size());if(!bin)fail("cannot write glTF buffer");
    std::ofstream(directory/"chain.gltf")<<j.dump(2)<<'\n';auto manifest=result_json(r);manifest["gltf"]="chain.gltf";
    for(size_t i=0;i<r.lods.size();++i){manifest["lods"][i]["gltf_mesh"]=j["nodes"][i]["mesh"];manifest["lods"][i]["gltf_node"]=i;}
    std::ofstream(directory/"lods.json")<<manifest.dump(2)<<'\n';
}
json settings_json(const Settings& s) {
    auto curve=[](const Curve& c){json a=json::array();for(auto p:c.points)a.push_back({p.x,p.y});return a;};
    auto views=[](ViewSet v){return json{{"orthographic",v.orthographic},{"perspective",v.perspective},{"seed",v.rotation_seed}};};
    return {{"levels",s.levels},{"output",s.output==OutputMode::Reuse?"reuse":"rebuild"},{"chain",s.chain==ChainMode::Direct?"direct":s.chain==ChainMode::Progressive?"progressive":"hybrid"},
      {"objective",s.objective==Objective::Quadric?"quadric":s.objective==Objective::Regularized?"regularized":s.objective==Objective::Visual?"visual":"topology_relaxed"},
      {"profile",s.profile==Profile::Coverage?"coverage":s.profile==Profile::Normals?"normals":"attributes"},{"pixels_per_meter",s.pixels_per_meter},
      {"meters_per_unit",s.meters_per_unit},{"base_pixels",s.base_pixels?json(*s.base_pixels):json(nullptr)},{"last_pixels",s.last_pixels},
      {"max_lod0_delta_px",s.max_lod0_delta_px?json(*s.max_lod0_delta_px):json(nullptr)},{"transition",curve(s.transition)},
      {"normal_importance",curve(s.normal_importance)},{"attribute_importance",curve(s.attribute_importance)},{"weights",{{"normal",s.weights.normal},{"color",s.weights.color},{"material",s.weights.material}}},
      {"search_views",views(s.search_views)},{"audit_views",views(s.audit_views)},{"search_supersample",s.search_supersample},{"audit_supersample",s.audit_supersample},
      {"max_supersample",s.max_supersample},{"candidate_budget",s.candidate_budget},{"beam_width",s.beam_width},{"prune",s.prune},{"force_scalar",s.force_scalar},{"coupled_wedges",s.coupled_wedges}};
}
Settings settings_json(const json& input) {
    Settings s;auto j=settings_json(s);for(auto it=input.begin();it!=input.end();++it){if(!j.contains(it.key()))fail("unknown setting: "+it.key());}
    j.merge_patch(input);
    auto mode=[&](const char* key,std::initializer_list<const char*> names){std::string v=j.at(key);unsigned i=0;for(auto name:names){if(v==name)return i;++i;}fail(std::string("unknown ")+key);};
    s.output=OutputMode(mode("output",{"rebuild","reuse"}));s.chain=ChainMode(mode("chain",{"direct","progressive","hybrid"}));s.profile=Profile(mode("profile",{"coverage","normals","attributes"}));s.objective=Objective(mode("objective",{"quadric","regularized","visual","topology_relaxed"}));
    auto byte=[&](const char* k){int n=j.at(k);if(n<0||n>255)fail("byte setting out of range");return uint8_t(n);};
    s.levels=byte("levels");s.beam_width=byte("beam_width");s.search_supersample=byte("search_supersample");s.audit_supersample=byte("audit_supersample");s.max_supersample=byte("max_supersample");
    int budget=j.at("candidate_budget");if(budget<1||budget>65535)fail("invalid candidate budget");s.candidate_budget=uint16_t(budget);
    s.pixels_per_meter=j.at("pixels_per_meter");s.meters_per_unit=j.at("meters_per_unit");s.last_pixels=j.at("last_pixels");
    if(j.contains("base_pixels")&&!j["base_pixels"].is_null())s.base_pixels=j["base_pixels"];
    if(j.contains("max_lod0_delta_px")&&!j["max_lod0_delta_px"].is_null())s.max_lod0_delta_px=j["max_lod0_delta_px"];
    auto curve=[&](const char* k,Curve& c){c.points.clear();for(auto& p:j.at(k)){if(p.size()!=2)fail("invalid curve point");c.points.push_back({p.at(0),p.at(1)});}};
    curve("transition",s.transition);curve("normal_importance",s.normal_importance);curve("attribute_importance",s.attribute_importance);
    s.weights={j["weights"].at("normal"),j["weights"].at("color"),j["weights"].at("material")};
    auto views=[&](const char* key){auto v=j.at(key);int o=v.at("orthographic"),p=v.at("perspective");if(o<0||p<0||o>65535||p>65535)fail("camera count out of range");return ViewSet{uint16_t(o),uint16_t(p),v.at("seed")};};
    s.search_views=views("search_views");s.audit_views=views("audit_views");s.prune=j.at("prune");s.force_scalar=j.at("force_scalar");s.coupled_wedges=j.at("coupled_wedges");
    if(auto e=validate(s);!e.empty())fail(e);return s;
}
}
