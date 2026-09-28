# Frontend multi-stage build: build React (Node 18), serve via nginx
FROM node:18-alpine AS build
WORKDIR /app
ARG REACT_APP_API_URL=quant.dileepkapu.com
ENV REACT_APP_API_URL=$REACT_APP_API_URL
ENV NODE_ENV=production
ENV DISABLE_ESLINT_PLUGIN=true
COPY package*.json ./
RUN npm install --legacy-peer-deps && npm install --save-dev typescript@4.9.5
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
# index.html is always re-checked (it names the current code files); the code files under
# /static/ carry a content hash in their names, so they can be cached for a year. Without
# this, browsers kept an old index.html after deploys and showed a blank page.
RUN printf 'server {\n\
  listen 80;\n\
  root /usr/share/nginx/html;\n\
  gzip on;\n\
  gzip_types text/css application/javascript application/json image/svg+xml;\n\
  location /static/ {\n\
    add_header Cache-Control "public, max-age=31536000, immutable";\n\
    try_files $uri =404;\n\
  }\n\
  location / {\n\
    add_header Cache-Control "no-cache";\n\
    try_files $uri /index.html;\n\
  }\n\
}\n' > /etc/nginx/conf.d/default.conf
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
