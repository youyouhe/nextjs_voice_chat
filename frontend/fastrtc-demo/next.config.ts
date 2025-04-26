import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  https: {
    cert: './localhost+2.pem',
    key: './localhost+2-key.pem',
  },
};

export default nextConfig;
