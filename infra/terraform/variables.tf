variable "name"              { default = "store-support-agent" }
variable "region"            { default = "us-east-1" }
variable "image_tag"         { default = "latest" }
variable "kill_switch"       { type = bool, default = false }   # terraform apply -var kill_switch=true
variable "allowed_cidr"      { default = "0.0.0.0/0" }          # narrow to your IP for the demo
variable "db_password"       { sensitive = true }
variable "token_signing_key" { sensitive = true }
variable "anthropic_api_key" { sensitive = true, default = "" }
variable "openai_api_key"    { sensitive = true, default = "" }
