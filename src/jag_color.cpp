#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
//#include <visualization_msgs/Marker.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_listener.h>
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"

#include <message_filters/time_synchronizer.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <image_geometry/pinhole_camera_model.h>
#include <image_transport/image_transport.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
//#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>

using namespace message_filters;
typedef pcl::PointCloud<pcl::PointXYZRGB> PCLCloud;
tf2_ros::Buffer tfbuffer;
ros::Publisher cloud_pub;
pcl::PCLPointCloud2::Ptr full_cloud(new pcl::PCLPointCloud2);

// GLOBAL VARIABLES
std::string img_topic_in, pc2_topic_in, pc2_topic_out;
std::string camera_info_topic;
std::string odom_frame, pc2_frame, img_frame;
int img_shift_x, img_shift_y, queue_size;
bool is_voxel;
double voxel_size;

void callback(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::CameraInfoPtr& info, const sensor_msgs::PointCloud2Ptr& cloud)
{
  image_geometry::PinholeCameraModel cam_model;
  cam_model.fromCameraInfo(info);
  cv_bridge::CvImagePtr input_bridge;

  cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
    cv_ptr = cv_bridge::toCvShare(image, "bgr8");
  }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  pcl::ExtractIndices<pcl::PCLPointCloud2> ext;
  
  PCLCloud pcl_cloud;
  pcl::PointCloud<pcl::PointXYZ> pc_xyz;
  pcl::fromROSMsg(*cloud, pc_xyz);
  pcl::copyPointCloud(pc_xyz, pcl_cloud);

  for(size_t i = 0; i < pcl_cloud.size() ; ++i)
  {
    geometry_msgs::PointStamped pt_velo, pt_usb;
    pt_velo.header.frame_id = pc2_frame;
    pt_velo.point.x = static_cast<double>(pc_xyz.points[i].x);
    pt_velo.point.y = static_cast<double>(pc_xyz.points[i].y);
    pt_velo.point.z = static_cast<double>(pc_xyz.points[i].z);

    try {
      tfbuffer.transform(pt_velo, pt_usb, img_frame);
    } catch (tf2::TransformException &ex) {
      ROS_WARN("%s",ex.what());
      ros::Duration(1.0).sleep();
    }

    cv::Point3d xyz(pt_usb.point.x,
                    pt_usb.point.y,
                    pt_usb.point.z);

    cv::Point2d imagePoint = cam_model.project3dToPixel(xyz);
//    ROS_INFO("h:%10.5f  w:%10.5f ", pt_usb.point.x, pt_usb.point.y);
    if(imagePoint.x > 0 && imagePoint.x < info->width
       && imagePoint.y > 0 && imagePoint.y < info->height
       && pt_usb.point.z > 0){    // Testing < and >
      int u = static_cast<int>(imagePoint.x);
      int v = static_cast<int>(imagePoint.y);
      cv::Vec3b colour = cv_ptr->image.at<cv::Vec3b>(v + img_shift_y, u + img_shift_x); // img offsets

      uchar b = colour.val[0];
      uchar g = colour.val[1];
      uchar r = colour.val[2];

//      ROS_INFO("r:%10.5d  g:%10.5d  b:%10.5d", r, g, b);
      pcl_cloud.points[i].r = r;
      pcl_cloud.points[i].g = g;
      pcl_cloud.points[i].b = b;
    }
    else {
      inliers->indices.push_back(static_cast<int>(i));
    }

  }  
  geometry_msgs::TransformStamped lid_odom;
  try {
    lid_odom = tfbuffer.lookupTransform(odom_frame, pc2_frame, cloud->header.stamp);
  } catch (tf2::TransformException &ex) {
    ROS_WARN("%s",ex.what());
    ros::Duration(0.1).sleep();
  }

  // Transform pc2 into odom frame
  tf::Transform lid_odom_tf;
  tf::transformMsgToTF(lid_odom.transform, lid_odom_tf);
  PCLCloud cloud_out;
  pcl_ros::transformPointCloud(pcl_cloud, cloud_out, lid_odom_tf);

  cloud_out.header.frame_id = odom_frame;

  // Obtain partial pc2 by filtering our parts without color
  pcl::PCLPointCloud2::Ptr pcl_partial(new pcl::PCLPointCloud2);
  pcl::toPCLPointCloud2(cloud_out, *pcl_partial);

  ext.setInputCloud(pcl_partial);
  ext.setIndices(inliers);
  ext.setNegative(true);
  ext.filter(*pcl_partial);

  // Apply voxelfilter to full cloud
  if(is_voxel == true){
    pcl::concatenatePointCloud(*full_cloud, *pcl_partial, *full_cloud);

    pcl::VoxelGrid<pcl::PCLPointCloud2> vg;
    vg.setInputCloud(full_cloud);
    vg.setLeafSize(voxel_size, voxel_size, voxel_size);
    vg.filter(*full_cloud);
  }

  // Apply statistical filter to full cloud
//  pcl::StatisticalOutlierRemoval<pcl::PCLPointCloud2> outlier;
//  outlier.setInputCloud(full_cloud);
//  outlier.setMeanK(50);
//  outlier.setStddevMulThresh(2.0);
//  outlier.filter(*full_cloud);

  if(is_voxel == true)
    cloud_pub.publish(*full_cloud);
  else
    cloud_pub.publish(*pcl_partial);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "jag_color");
  ros::NodeHandle nh("~");

  nh.getParam("img_topic_in", img_topic_in);
  nh.getParam("pc2_topic_in", pc2_topic_in);
  nh.getParam("pc2_topic_out", pc2_topic_out);
  nh.getParam("camera_info_topic", camera_info_topic);

  nh.getParam("odom_frame", odom_frame);
  nh.getParam("pc2_frame", pc2_frame);
  nh.getParam("img_frame", img_frame);

  nh.getParam("img_shift_x", img_shift_x);
  nh.getParam("img_shift_y", img_shift_y);
  nh.getParam("queue_size", queue_size);

  nh.getParam("is_voxel", is_voxel);
  nh.getParam("voxel_size", voxel_size);

  image_transport::ImageTransport it(nh);
  tf2_ros::TransformListener tf_listener(tfbuffer);
  cloud_pub = nh.advertise<sensor_msgs::PointCloud2> (pc2_topic_out, 1);
  message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, img_topic_in, 1);
  message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub(nh, camera_info_topic, 1);
  message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub(nh, pc2_topic_in, 1);
  typedef sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::PointCloud2> SyncPolicy;
  Synchronizer<SyncPolicy> sync(SyncPolicy(queue_size), image_sub, info_sub, cloud_sub); // Changed from 2 to 10
  sync.registerCallback(callback);
  ros::spin();
  pcl::PCDWriter obj;
  obj.writeBinary("cloud.pcd", *full_cloud);
}
