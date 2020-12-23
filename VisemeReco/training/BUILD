load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

cc_binary(
    name = "training_data",
    srcs = ["training_data.cc"],
   data = glob([
        "data/**/*",
    ]),

    deps = [
        "//third_party/libSVM:libSVM",
        ":mfcc5_cc_proto",
        "@com_google_absl//absl/memory",
    ],
)

cc_proto_library(
    name = "mfcc5_cc_proto",
    deps = [":mfcc5_proto"],
)

proto_library(
    name = "mfcc5_proto",
    srcs = ["mfcc5.proto"],
    visibility = ["//visibility:public"],
)

