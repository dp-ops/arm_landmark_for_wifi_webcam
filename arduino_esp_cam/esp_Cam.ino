#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

#define CAMERA_MODEL_AI_THINKER //for camera pins
#include "camera_pins.h"

const char* ssid = "Tsinakos_2.4G";
const char* password = "OriginalMastoura";

httpd_handle_t stream_httpd = NULL; //create the server to broadcast

// Copied most of it from ESP32->Example->CameraWebServer something
static esp_err_t stream_handler(httpd_req_t *req){
    camera_fb_t * fb = NULL;
    esp_err_t res = ESP_OK;
    size_t _jpg_buf_len = 0;
    uint8_t * _jpg_buf = NULL;
    char * part_buf[64];

    static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=123456789000000000000987654321";
    static const char* _STREAM_BOUNDARY = "\r\n--123456789000000000000987654321\r\n";
    static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if(res != ESP_OK){
        return res;
    }

    // Capture loop
    while(true){
        fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("Camera capture failed");
            res = ESP_FAIL;
        } else {
            if(fb->format != PIXFORMAT_JPEG){
                Serial.println("Frame is not JPEG, cannot stream.");
                res = ESP_FAIL;
            } else {
                _jpg_buf_len = fb->len;
                _jpg_buf = fb->buf;
            }
        }

        if(res == ESP_OK){
            size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, _jpg_buf_len);
            res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
        }
        if(res == ESP_OK){
            res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
        }
        if(res == ESP_OK){
            res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        }

        // Always return the frame buffer to be reused
        if(fb){
            esp_camera_fb_return(fb);
            fb = NULL;
            _jpg_buf = NULL;
        } else if(_jpg_buf){
            free(_jpg_buf);
            _jpg_buf = NULL;
        }

        // Break if there's an error in sending
        if(res != ESP_OK){
            break;
        }
    }
    return res;
}

//configure and initialize the camera
esp_err_t config_camera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // ------Quality specs: frame size, quality, and frame count----
  config.frame_size = FRAMESIZE_VGA;   // SVGA: 800x600, VGA: 600x600 nomizw
  config.jpeg_quality = 40;            // 0-63, lower is better.
  config.fb_count = 2;                 // Two buffers for better performance

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return err;
  }

  sensor_t * s = esp_camera_sensor_get();
  // Check if we need to rotate/flip (OV2640 needs this based on orientation)
  if (s->id.PID == OV2640_PID) {
    s->set_vflip(s, 1);       // ENABLE Vertical Flip (from image)
    s->set_hmirror(s, 1);     // ENABLE Horizontal Mirror (from image)
    // The image had special effect to No effect and brightness/contrast/saturation to 0, which are defaults
  }
  
  // set quality and framesize again to be sure
  s->set_framesize(s, config.frame_size);
  s->set_quality(s, config.jpeg_quality);
  
  // AEC, AGC, and AWB were all "Yes" in the image for better quality in changing env.
  s->set_exposure_ctrl(s, 1); // Enable Auto Exposure Control
  s->set_gain_ctrl(s, 1);     // Enable Auto Gain Control (this one was already correct)
  s->set_whitebal(s, 1);      // Enable Auto White Balance

  //Black and white
  s->set_special_effect(s, 0); //Den xreiazetai kanonika, giati einai effect, h eikona feygei RGB apo thn kamera

  Serial.println("Camera configured.");
  return ESP_OK;
}

void startCameraServer(){
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;

    httpd_uri_t stream_uri = {
        .uri       = "/stream",
        .method    = HTTP_GET,
        .handler   = stream_handler,
        .user_ctx  = NULL
    };
    
    Serial.printf("Starting stream server on port: '%d'\n", config.server_port);
    if (httpd_start(&stream_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(stream_httpd, &stream_uri);
    }
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  // 1. Initialize the camera 
  if(config_camera() != ESP_OK){
      Serial.println("Could not initialize camera. Power cycle the board and try again.");
      while(true) delay(1);
  }

  // 2. Connect to Wi-Fi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWi-Fi connected");

  // 3. Start the stream server
  startCameraServer();

  // 4. Print the URL to use in your Python script
  Serial.print("Camera stream ready! Use this URL in Python: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/stream");
}

void loop() {
  // Nothing to do in the main loop
  delay(1);
}