#pragma once
#include "neural_data.hpp"
#include "neural_vertex_storage.hpp"
namespace blitz::neural::training {
// Compact working geometry and an immutable FP32 audit reference have separate
// files. Never use the quantized copy as its own visual ground truth.
struct PreparedMesh {
    Mesh source,draw;VertexBounds quantization;json provenance;
};
inline void write_reference(std::ostream& f,const Mesh& m){
    f.write("BLZMREF1",8);write_vector(f,m.positions);write_vector(f,m.normals);write_vector(f,m.uv);write_vector(f,m.colors);write_vector(f,m.tangents);write_vector(f,m.indices);write_vector(f,m.materials);write_vector(f,m.double_sided);
}
inline Mesh read_reference(std::istream& f){
    char magic[8];f.read(magic,8);if(!f||std::memcmp(magic,"BLZMREF1",8))throw std::invalid_argument("mesh reference version");Mesh m;constexpr size_t vertices=40000000,faces=80000000;
    m.positions=read_vector<Vec3>(f,vertices);m.normals=read_vector<Vec3>(f,vertices);m.uv=read_vector<Vec2>(f,vertices);m.colors=read_vector<ColorRGBA8>(f,vertices);m.tangents=read_vector<Vec4>(f,vertices);m.indices=read_vector<uint32_t>(f,faces*3);m.materials=read_vector<uint16_t>(f,faces);m.double_sided=read_vector<uint8_t>(f,65536);
    if(f.peek()!=EOF)throw std::invalid_argument("mesh reference trailing data");if(auto error=validate(m.view());!error.empty())throw std::invalid_argument(error);return m;
}
inline void write_packed_mesh(std::ostream& f,MeshView m){
    if(auto error=validate(m);!error.empty())throw std::invalid_argument(error);auto q=vertex_bounds(m);std::vector<uint16_t> positions,uv;std::vector<uint32_t> normals,tangents;std::vector<ColorRGBA8> colors;
    for(size_t i=0;i<m.positions.count;++i){auto p=m.positions[i];positions.insert(positions.end(),{pack_unorm16(p.x,q.low.x,q.extent.x),pack_unorm16(p.y,q.low.y,q.extent.y),pack_unorm16(p.z,q.low.z,q.extent.z)});}
    for(size_t i=0;i<m.normals.count;++i)normals.push_back(pack_direction(m.normals[i]));
    for(size_t i=0;i<m.uv.count;++i){auto v=m.uv[i];if(v.x< -8||v.x>8||v.y< -8||v.y>8)throw std::invalid_argument("packed cache UV outside [-8,8]");uv.insert(uv.end(),{pack_unorm16(v.x,-8,16),pack_unorm16(v.y,-8,16)});}
    for(size_t i=0;i<m.colors.count;++i)colors.push_back(m.colors[i]);
    for(size_t i=0;i<m.tangents.count;++i){auto t=m.tangents[i];tangents.push_back(pack_direction({t.x,t.y,t.z},t.w<0?-1:1));}
    f.write("BLZMPK01",8);f.write(reinterpret_cast<const char*>(&q),sizeof(q));write_vector(f,positions);write_vector(f,normals);write_vector(f,uv);write_vector(f,colors);write_vector(f,tangents);
    write_vector(f,std::vector<uint32_t>(m.indices.begin(),m.indices.end()));write_vector(f,std::vector<uint16_t>(m.materials.begin(),m.materials.end()));write_vector(f,std::vector<uint8_t>(m.double_sided.begin(),m.double_sided.end()));
}
inline std::pair<Mesh,VertexBounds> read_packed_mesh(std::istream& f){
    char magic[8];VertexBounds q;f.read(magic,8);f.read(reinterpret_cast<char*>(&q),sizeof(q));if(!f||std::memcmp(magic,"BLZMPK01",8))throw std::invalid_argument("packed mesh version");
    for(float v:{q.low.x,q.low.y,q.low.z,q.extent.x,q.extent.y,q.extent.z})if(!std::isfinite(v))throw std::invalid_argument("nonfinite packed bounds");if(q.extent.x<0||q.extent.y<0||q.extent.z<0)throw std::invalid_argument("negative packed extent");
    constexpr size_t vertices=40000000,faces=80000000;auto p=read_vector<uint16_t>(f,vertices*3);auto normal=read_vector<uint32_t>(f,vertices);auto uv=read_vector<uint16_t>(f,vertices*2);auto colors=read_vector<ColorRGBA8>(f,vertices);auto tangent=read_vector<uint32_t>(f,vertices);
    size_t n=p.size()/3;if(!n||p.size()%3||(!normal.empty()&&normal.size()!=n)||(!uv.empty()&&uv.size()!=n*2)||(!colors.empty()&&colors.size()!=n)||(!tangent.empty()&&tangent.size()!=n))throw std::invalid_argument("packed mesh dimensions");Mesh m;
    m.positions.reserve(n);for(size_t i=0;i<n;++i)m.positions.push_back({unpack_unorm16(p[i*3],q.low.x,q.extent.x),unpack_unorm16(p[i*3+1],q.low.y,q.extent.y),unpack_unorm16(p[i*3+2],q.low.z,q.extent.z)});
    for(auto v:normal)m.normals.push_back(unpack_direction(v));for(size_t i=0;i<uv.size();i+=2)m.uv.push_back({unpack_unorm16(uv[i],-8,16),unpack_unorm16(uv[i+1],-8,16)});m.colors=std::move(colors);
    for(auto v:tangent){auto t=unpack_direction(v);auto a=v>>30;if(a!=1&&a!=3)throw std::invalid_argument("packed tangent sign");m.tangents.push_back({t.x,t.y,t.z,a==3?-1.f:1.f});}
    m.indices=read_vector<uint32_t>(f,faces*3);m.materials=read_vector<uint16_t>(f,faces);m.double_sided=read_vector<uint8_t>(f,65536);if(f.peek()!=EOF)throw std::invalid_argument("packed mesh trailing data");if(auto error=validate(m.view());!error.empty())throw std::invalid_argument(error);return {std::move(m),q};
}
inline PreparedMesh prepared_mesh(const json& metadata,const fs::path& cache){
    auto key=metadata.dump();auto directory=cache/sha256(std::as_bytes(std::span(key)));auto index=directory/"index.json";PreparedMesh result;
    if(!fs::exists(index)){
        result.source=load_mesh(metadata.at("path").get<std::string>());fs::create_directories(directory);
        auto save=[&](const char* name,auto write){auto path=directory/name,temp=path;temp+=".part";{std::ofstream f(temp,std::ios::binary);write(f);f.close();if(!f)throw std::runtime_error("prepared mesh write failed");}fs::rename(temp,path);};
        save("reference.bin",[&](auto& f){write_reference(f,result.source);});save("packed.bin",[&](auto& f){write_packed_mesh(f,result.source.view());});
        write_json(index,{{"version",1},{"asset",metadata},{"reference_sha256",file_sha256(directory/"reference.bin")},{"packed_sha256",file_sha256(directory/"packed.bin")},{"reference_bytes",fs::file_size(directory/"reference.bin")},{"packed_bytes",fs::file_size(directory/"packed.bin")}});
    }
    auto provenance=read_json(index);if(provenance.at("version")!=1||provenance.at("asset")!=metadata||provenance.at("reference_sha256")!=file_sha256(directory/"reference.bin")||provenance.at("packed_sha256")!=file_sha256(directory/"packed.bin"))throw std::invalid_argument("prepared mesh checksum/contract changed");
    if(result.source.positions.empty()){std::ifstream f(directory/"reference.bin",std::ios::binary);result.source=read_reference(f);}std::ifstream f(directory/"packed.bin",std::ios::binary);auto [mesh,q]=read_packed_mesh(f);result.draw=std::move(mesh);result.quantization=q;result.provenance=provenance;return result;
}
}
