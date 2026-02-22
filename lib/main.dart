import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:cunning_document_scanner/cunning_document_scanner.dart';
import 'package:edge_detection/edge_detection.dart';
import 'package:image_picker/image_picker.dart';
import 'package:easy_image_viewer/easy_image_viewer.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'dart:io';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Document Scanner',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
      ),
      home: const MyHomePage(title: 'Document Scanner MVP'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});
  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final List<String> _scannedPages = [];
  bool _isScanning = false;

  Future<String?> _scanWithCunning() async {
    try {
      final pictures = await CunningDocumentScanner.getPictures(
        noOfPages: 1,
        isGalleryImportAllowed: true,
      );
      if (pictures != null && pictures.isNotEmpty) {
        return pictures.first;
      }
    } catch (_) {}
    return null;
  }

  Future<String?> _scanWithEdgeDetection() async {
    try {
      final dir = await getTemporaryDirectory();
      final outPath = p.join(
        dir.path,
        'scan_${DateTime.now().millisecondsSinceEpoch}.jpg',
      );
      final ok = await EdgeDetection.detectEdge(
        outPath,
        canUseGallery: true,
        androidScanTitle: 'Scan Dokumen',
        androidCropTitle: 'Sesuaikan Area',
        androidCropBlackWhiteTitle: 'Hitam Putih',
        androidCropReset: 'Reset',
      );
      if (ok && await File(outPath).exists()) {
        return outPath;
      }
    } catch (_) {}
    return null;
  }

  Future<String?> _scanWithLegacyOpenCV() async {
    try {
      final picker = ImagePicker();
      final img = await picker.pickImage(source: ImageSource.camera);
      if (img == null) return null;
      return await OpenCVBridge.autoCrop(img.path);
    } catch (_) {
      return null;
    }
  }

  Future<String?> _runAutoCrop() async {
    final camStatus = await Permission.camera.request();
    if (!camStatus.isGranted) {
      if (!mounted) return null;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Izin kamera dibutuhkan untuk scan')),
      );
      return null;
    }

    String? resultPath = await _scanWithCunning();
    resultPath ??= await _scanWithEdgeDetection();
    resultPath ??= await _scanWithLegacyOpenCV();

    if (!mounted) return null;

    if (resultPath == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Scan gagal di semua mode')),
      );
      return null;
    }
    return resultPath;
  }

  Future<void> _addScannedPage() async {
    if (_isScanning) return;
    setState(() => _isScanning = true);
    try {
      final resultPath = await _runAutoCrop();
      if (!mounted || resultPath == null) return;
      setState(() => _scannedPages.add(resultPath));
      _previewImage(resultPath);
    } finally {
      if (mounted) setState(() => _isScanning = false);
    }
  }

  Future<void> _scanWithLegacyFromGallery() async {
    final picker = ImagePicker();
    final img = await picker.pickImage(source: ImageSource.gallery);
    if (img == null) return;
    try {
      final result = await OpenCVBridge.autoCrop(img.path);
      if (!mounted) return;
      setState(() => _scannedPages.add(result));
      _previewImage(result);
    } catch (_) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Auto crop dari galeri gagal')),
      );
    }
  }

  void _previewImage(String path) {
    showImageViewer(
      context,
      FileImage(File(path)),
      swipeDismissible: true,
      doubleTapZoomable: true,
    );
  }

  void _removePage(String path) {
    setState(() => _scannedPages.remove(path));
  }

  void _movePage(int oldIndex, int newIndex) {
    setState(() {
      if (newIndex > oldIndex) newIndex -= 1;
      final item = _scannedPages.removeAt(oldIndex);
      _scannedPages.insert(newIndex, item);
    });
  }

  void _showMvpInfo() {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text(
          'MVP saat ini: scan multi-page, preview, hapus, reorder. Next: filter/rotate + export PDF + OCR.',
        ),
        duration: Duration(seconds: 3),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
        actions: [
          IconButton(
            onPressed: _showMvpInfo,
            icon: const Icon(Icons.info_outline),
            tooltip: 'MVP status',
          ),
        ],
      ),
      body: _scannedPages.isEmpty
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.document_scanner_outlined,
                      size: 64,
                      color: Theme.of(context).colorScheme.primary,
                    ),
                    const SizedBox(height: 12),
                    const Text(
                      'Belum ada halaman scan',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 6),
                    const Text(
                      'Tambahkan halaman dari kamera untuk menguji autocrop dan susun dokumen multi-page.',
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 16),
                    FilledButton.icon(
                      onPressed: _isScanning ? null : _addScannedPage,
                      icon: _isScanning
                          ? const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.camera_alt),
                      label: Text(
                        _isScanning ? 'Scanning...' : 'Scan Halaman',
                      ),
                    ),
                  ],
                ),
              ),
            )
          : ReorderableListView.builder(
              padding: const EdgeInsets.fromLTRB(12, 12, 12, 120),
              itemCount: _scannedPages.length,
              onReorder: _movePage,
              itemBuilder: (context, index) {
                final path = _scannedPages[index];
                return Card(
                  key: ValueKey(path),
                  child: ListTile(
                    leading: ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: Image.file(
                        File(path),
                        width: 52,
                        height: 52,
                        fit: BoxFit.cover,
                        errorBuilder: (context, error, stackTrace) {
                          return Container(
                            width: 52,
                            height: 52,
                            color: Colors.grey.shade200,
                            alignment: Alignment.center,
                            child: const Icon(Icons.broken_image_outlined),
                          );
                        },
                      ),
                    ),
                    title: Text('Halaman ${index + 1}'),
                    subtitle: Text(
                      p.basename(path),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    onTap: () => _previewImage(path),
                    trailing: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        IconButton(
                          tooltip: 'Preview',
                          onPressed: () => _previewImage(path),
                          icon: const Icon(Icons.visibility_outlined),
                        ),
                        IconButton(
                          tooltip: 'Hapus',
                          onPressed: () => _removePage(path),
                          icon: const Icon(Icons.delete_outline),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
      floatingActionButton: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          FloatingActionButton.extended(
            heroTag: 'scan_page',
            onPressed: _isScanning ? null : _addScannedPage,
            icon: _isScanning
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.document_scanner),
            label: const Text('Scan'),
          ),
          const SizedBox(height: 12),
          FloatingActionButton.small(
            heroTag: 'gallery_crop',
            onPressed: _scanWithLegacyFromGallery,
            tooltip: 'Auto crop dari galeri (OpenCV)',
            child: const Icon(Icons.photo_library_outlined),
          ),
          const SizedBox(height: 12),
          FloatingActionButton.small(
            heroTag: 'test_opencv',
            onPressed: () async {
              final messenger = ScaffoldMessenger.of(context);
              final result = await OpenCVBridge.test();
              if (!mounted) return;
              messenger.showSnackBar(SnackBar(content: Text('OpenCV: $result')));
            },
            tooltip: 'Test OpenCV',
            child: const Icon(Icons.science_outlined),
          ),
        ],
      ),
      bottomNavigationBar: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(12, 8, 12, 12),
          child: Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _scannedPages.isEmpty ? null : _showMvpInfo,
                  icon: const Icon(Icons.tune),
                  label: const Text('Filter/Rotate (next)'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: FilledButton.icon(
                  onPressed: _scannedPages.isEmpty
                      ? null
                      : () {
                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(
                              content: Text(
                                'Siap export PDF (${_scannedPages.length} halaman). Tambahkan paket pdf/printing untuk finalisasi.',
                              ),
                            ),
                          );
                        },
                  icon: const Icon(Icons.picture_as_pdf_outlined),
                  label: const Text('Export PDF'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class OpenCVBridge {
  static const _channel = MethodChannel("opencv_channel");

  static Future<String> test() async {
    return await _channel.invokeMethod("testOpenCV");
  }

  static Future<String> detectEdges(String path) async {
    // Menggunakan path yang diberikan, bukan mengambil gambar lagi
    if (path.isNotEmpty) {
      // Panggil platform channel untuk memproses edge detection di sisi native/Kotlin/Swift
      final edgePath = await _channel.invokeMethod<String>("detectEdges", {
        "path": path,
      });
      if (edgePath != null) {
        return edgePath;
      } else {
        throw Exception("Edge detection failed: result is null");
      }
    } else {
      throw Exception("Path gambar tidak boleh kosong");
    }
  }

  static Future<String> autoCrop(String path) async {
    return await _channel.invokeMethod("autoCrop", {"path": path});
  }
}
