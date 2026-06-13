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
RUN printf 'server {\n\
  listen 80;\n\
  root /usr/share/nginx/html;\n\
  index index.html;\n\
  location / { try_files $uri $uri/ /index.html; }\n\
}\n' > /etc/nginx/conf.d/default.conf
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
