import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:typed_data';

void main() {
  runApp(RetinaSegmentationApp());
}

class RetinaSegmentationApp extends StatefulWidget {
  @override
  _RetinaSegmentationAppState createState() => _RetinaSegmentationAppState();
}

class _RetinaSegmentationAppState extends State<RetinaSegmentationApp> {
  File? _retinaImage;
  File? _maskImage;
  Uint8List? _segmentedImage;
  final ImagePicker _picker = ImagePicker();

  // Pick an image from gallery
  Future<void> _pickImage(bool isRetina) async {
    final pickedFile = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        if (isRetina) {
          _retinaImage = File(pickedFile.path);
        } else {
          _maskImage = File(pickedFile.path);
        }
      });
    }
  }

  // Upload images and get segmented result
  Future<void> _uploadAndSegment() async {
    if (_retinaImage == null || _maskImage == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Please select both images!")),
      );
      return;
    }

    var request = http.MultipartRequest(
      'POST',
      Uri.parse("https://retina-segmentation.onrender.com/segment"), // ✅ Replace with your Render URL
    );

    request.files.add(await http.MultipartFile.fromPath('retina_image', _retinaImage!.path));
    request.files.add(await http.MultipartFile.fromPath('mask_image', _maskImage!.path));

    var response = await request.send();
    var responseBody = await response.stream.bytesToString();

    if (response.statusCode == 200) {
      var jsonResponse = jsonDecode(responseBody);
      String base64String = jsonResponse["segmented_image"];

      setState(() {
        _segmentedImage = base64Decode(base64String);
      });
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error in segmentation!")),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: Text("Retina Blood Vessel Segmentation")),
        body: SingleChildScrollView(
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _retinaImage == null
                    ? Text("Select Retina Image")
                    : Image.file(_retinaImage!, width: 150, height: 150),
                ElevatedButton(
                  onPressed: () => _pickImage(true),
                  child: Text("Pick Retina Image"),
                ),
                _maskImage == null
                    ? Text("Select Mask Image")
                    : Image.file(_maskImage!, width: 150, height: 150),
                ElevatedButton(
                  onPressed: () => _pickImage(false),
                  child: Text("Pick Mask Image"),
                ),
                SizedBox(height: 20),
                ElevatedButton(
                  onPressed: _uploadAndSegment,
                  child: Text("Segment Image"),
                ),
                SizedBox(height: 20),

                // Ensuring the segmented image has proper spacing
                _segmentedImage == null
                    ? Text("Segmented Image will appear here")
                    : Padding(
                  padding: EdgeInsets.only(bottom: 50), // Added space below
                  child: Container(
                    width: 300,
                    height: 300,
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.black),
                    ),
                    child: Image.memory(
                      _segmentedImage!,
                      width: 300,
                      height: 300,
                      fit: BoxFit.contain, // Keeps aspect ratio
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
