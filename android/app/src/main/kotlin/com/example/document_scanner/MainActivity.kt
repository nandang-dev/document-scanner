package com.example.document_scanner

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.io.File
import java.util.concurrent.Executors

import kotlin.math.acos
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class MainActivity : FlutterActivity() {

    private val CHANNEL = "opencv_channel"
    private val executor = Executors.newSingleThreadExecutor()

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        // Load OpenCV
        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "OpenCV failed")
        } else {
            Log.d("OpenCV", "OpenCV loaded")
        }

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "testOpenCV" -> {
                    result.success("OpenCV Ready")
                }

                "detectEdges" -> {
                    val path = call.argument<String>("path")
                    if (path.isNullOrEmpty()) {
                        result.error("INVALID_ARGS", "Path cannot be null or empty", null)
                    } else {
                        executor.execute {
                            try {
                                result.success(detectEdges(path))
                            } catch (e: Exception) {
                                result.error("EDGE_DETECTION_FAILED", e.message, null)
                            }
                        }
                    }
                }

                "autoCrop" -> {
                    val path = call.argument<String>("path")
                    if (path.isNullOrEmpty()) {
                        result.error("INVALID_ARGS", "Path cannot be null or empty", null)
                    } else {
                        executor.execute {
                            try {
                                result.success(autoCrop(path))
                            } catch (e: Exception) {
                                result.error("AUTO_CROP_FAILED", e.message, null)
                            }
                        }
                    }
                }

                else -> {
                    result.notImplemented()
                }
            }
        }
    }

    private fun detectEdges(path: String): String {

        val bitmap = BitmapFactory.decodeFile(path)
            ?: throw IllegalArgumentException("Failed to decode image at path: $path")
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)

        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)

        val blur = Mat()
        Imgproc.GaussianBlur(gray, blur, Size(5.0,5.0),0.0)

        val edges = Mat()
        Imgproc.Canny(blur, edges, 75.0, 200.0)

        val resultBitmap =
            Bitmap.createBitmap(edges.cols(), edges.rows(), Bitmap.Config.ARGB_8888)

        Utils.matToBitmap(edges, resultBitmap)

        val outFile = File(cacheDir, "edges_${System.currentTimeMillis()}.png")
        outFile.outputStream().use {
            resultBitmap.compress(Bitmap.CompressFormat.PNG, 100, it)
        }

        return outFile.absolutePath
    }

    private fun autoCrop(path: String): String {

        val bitmap = BitmapFactory.decodeFile(path)
            ?: throw IllegalArgumentException("Failed to decode image at path: $path")
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)

        // Downscale untuk deteksi - gambar besar punya terlalu banyak detail
        val maxDim = 1200
        val scale = min(1.0, maxDim.toDouble() / max(src.rows(), src.cols()))
        val detectSrc = if (scale < 0.99) {
            val resized = Mat()
            Imgproc.resize(src, resized, Size(), scale, scale, Imgproc.INTER_AREA)
            resized
        } else src

        val imageArea = detectSrc.rows() * detectSrc.cols()
        val minArea = imageArea * 0.015  // 1.5% - support smaller docs like KTP
        // Kertas sering memenuhi hampir seluruh frame, jadi jangan terlalu agresif membuang area besar.
        val maxArea = imageArea * 0.995

        var doc: MatOfPoint2f? = findDocumentContour(detectSrc, minArea, maxArea)

        if (doc == null) doc = findDocumentContourAdaptive(detectSrc, minArea, maxArea)
        if (doc == null) doc = findDocumentContourOtsu(detectSrc, minArea, maxArea)
        if (doc == null) doc = findDocumentByGrabCut(detectSrc, minArea, maxArea)
        if (doc == null) doc = findDocumentRectFallback(detectSrc, minArea, maxArea)

        if (doc == null) {
            val roi = findBestDocumentRect(detectSrc, minArea)
            if (roi != null) {
                val scaled = if (scale < 0.99) {
                    Rect(
                        (roi.x / scale).toInt().coerceIn(0, src.cols() - 1),
                        (roi.y / scale).toInt().coerceIn(0, src.rows() - 1),
                        (roi.width / scale).toInt().coerceAtLeast(1),
                        (roi.height / scale).toInt().coerceAtLeast(1)
                    )
                } else roi
                val safe = clampRectToMat(scaled, src.cols(), src.rows())
                val cropped = Mat(src, safe).clone()
                return saveCrop(cropped)
            }
            val inset = centerInsetRect(src.cols(), src.rows(), 0.02)
            return saveCrop(Mat(src, inset).clone())
        }
        doc = refineQuadCorners(detectSrc, doc)

        // Scale points back ke ukuran gambar asli
        val ptsUnclamped = if (scale < 0.99) {
            doc.toArray().map { Point(it.x / scale, it.y / scale) }.toTypedArray()
        } else doc.toArray()

        val pts = ptsUnclamped.map {
            Point(
                it.x.coerceIn(0.0, (src.cols() - 1).toDouble()),
                it.y.coerceIn(0.0, (src.rows() - 1).toDouble())
            )
        }.toTypedArray()
        val ordered = orderPoints(pts)
        val tl = ordered[0]
        val tr = ordered[1]
        val br = ordered[2]
        val bl = ordered[3]

        val widthA = distance(br, bl)
        val widthB = distance(tr, tl)
        val maxWidth = max(widthA, widthB).toInt().coerceAtLeast(1)

        val heightA = distance(tr, br)
        val heightB = distance(tl, bl)
        val maxHeight = max(heightA, heightB).toInt().coerceAtLeast(1)

        val srcPts = MatOfPoint2f(tl, tr, br, bl)
        val dstPts = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(maxWidth - 1.0, 0.0),
            Point(maxWidth - 1.0, maxHeight - 1.0),
            Point(0.0, maxHeight - 1.0)
        )

        val M = Imgproc.getPerspectiveTransform(srcPts, dstPts)
        var warp = Mat()
        Imgproc.warpPerspective(src, warp, M, Size(maxWidth.toDouble(), maxHeight.toDouble()))
        warp = deskewByHough(warp)
        warp = cropToContent(warp)
        return saveCrop(if (warp.empty()) src else warp)
    }

    private fun findDocumentContour(src: Mat, minArea: Double, maxArea: Double): MatOfPoint2f? {
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(gray, gray)

        val blur = Mat()
        Imgproc.GaussianBlur(gray, blur, Size(5.0, 5.0), 0.0)

        val edged = Mat()
        Imgproc.Canny(blur, edged, 50.0, 150.0)

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        Imgproc.morphologyEx(edged, edged, Imgproc.MORPH_CLOSE, kernel)

        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(
            edged, contours, Mat(),
            Imgproc.RETR_LIST,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        contours.sortByDescending { Imgproc.contourArea(it) }
        return findBestQuadFromContours(contours, minArea, maxArea, src.rows(), src.cols())
    }

    private fun findDocumentContourAdaptive(src: Mat, minArea: Double, maxArea: Double): MatOfPoint2f? {
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(gray, gray)

        val blur = Mat()
        Imgproc.GaussianBlur(gray, blur, Size(5.0, 5.0), 0.0)

        val thresh = Mat()
        Imgproc.adaptiveThreshold(
            blur, thresh, 255.0,
            Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
            Imgproc.THRESH_BINARY,
            11, 2.0
        )

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, kernel)

        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(
            thresh, contours, Mat(),
            Imgproc.RETR_LIST,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        contours.sortByDescending { Imgproc.contourArea(it) }
        return findBestQuadFromContours(contours, minArea, maxArea, src.rows(), src.cols())
    }

    private fun findDocumentContourOtsu(src: Mat, minArea: Double, maxArea: Double): MatOfPoint2f? {
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(gray, gray)

        val blur = Mat()
        Imgproc.GaussianBlur(gray, blur, Size(5.0, 5.0), 0.0)

        val thresh = Mat()
        Imgproc.threshold(blur, thresh, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, kernel)

        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(
            thresh, contours, Mat(),
            Imgproc.RETR_LIST,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        contours.sortByDescending { Imgproc.contourArea(it) }
        var result = findBestQuadFromContours(contours, minArea, maxArea, src.rows(), src.cols())

        // Coba inversi jika dokumen lebih gelap dari background
        if (result == null) {
            Imgproc.threshold(blur, thresh, 0.0, 255.0, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU)
            Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, kernel)
            contours.clear()
            Imgproc.findContours(thresh, contours, Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
            contours.sortByDescending { Imgproc.contourArea(it) }
            result = findBestQuadFromContours(contours, minArea, maxArea, src.rows(), src.cols())
        }
        return result
    }

    private fun findBestQuadFromContours(
        contours: List<MatOfPoint>,
        minArea: Double,
        maxArea: Double,
        rows: Int,
        cols: Int
    ): MatOfPoint2f? {
        val epsFactors = listOf(0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15)
        var bestQuad: MatOfPoint2f? = null
        var bestScore = Double.NEGATIVE_INFINITY

        for (c in contours) {
            val area = Imgproc.contourArea(c)
            if (area > maxArea) continue
            if (area < minArea) break

            // Try direct approxPolyDP first
            for (epsFactor in epsFactors) {
                val approx = approxContourToQuad(c, epsFactor)
                if (approx != null) {
                    val score = scoreQuad(approx, rows, cols)
                    if (score > bestScore) {
                        bestScore = score
                        bestQuad = approx
                    }
                }
            }

            // Fallback: convex hull - untuk dokumen dengan banyak konten (teks, tanda tangan, stempel)
            // hull mengabaikan detail internal dan mengambil outline luar dokumen
            val hull = MatOfInt()
            Imgproc.convexHull(c, hull)
            val contourArray = c.toArray()
            val hullPointsList = mutableListOf<Point>()
            for (i in 0 until hull.rows()) {
                val idx = hull.get(i, 0)[0].toInt()
                hullPointsList.add(contourArray[idx])
            }
            val hullMatOfPoint2f = MatOfPoint2f(*hullPointsList.toTypedArray())
            for (epsFactor in epsFactors) {
                val peri = Imgproc.arcLength(hullMatOfPoint2f, true)
                val approx = MatOfPoint2f()
                Imgproc.approxPolyDP(hullMatOfPoint2f, approx, epsFactor * peri, true)
                if (approx.total().toInt() == 4) {
                    val score = scoreQuad(approx, rows, cols)
                    if (score > bestScore) {
                        bestScore = score
                        bestQuad = approx
                    }
                }
            }
        }
        return if (bestScore >= 0.46) bestQuad else null
    }

    private fun findDocumentRectFallback(src: Mat, minArea: Double, maxArea: Double): MatOfPoint2f? {
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(gray, gray)
        val blur = Mat()
        Imgproc.GaussianBlur(gray, blur, Size(5.0, 5.0), 0.0)
        val edges = Mat()
        Imgproc.Canny(blur, edges, 50.0, 150.0)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, kernel)

        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(edges, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        contours.sortByDescending { Imgproc.contourArea(it) }

        var bestQuad: MatOfPoint2f? = null
        var bestScore = Double.NEGATIVE_INFINITY
        for (c in contours) {
            val contourArea = Imgproc.contourArea(c)
            if (contourArea > maxArea) continue
            if (contourArea < minArea) break

            val rr = Imgproc.minAreaRect(MatOfPoint2f(*c.toArray()))
            val rectArea = rr.size.area()
            if (rectArea <= 1.0) continue

            val fill = contourArea / rectArea
            if (fill < 0.65) continue

            val points = arrayOf(Point(), Point(), Point(), Point())
            rr.points(points)
            val quad = MatOfPoint2f(*points)
            val score = scoreQuad(quad, src.rows(), src.cols()) + fill * 0.25
            if (score > bestScore) {
                bestScore = score
                bestQuad = quad
            }
        }
        return bestQuad
    }

    private fun findDocumentByGrabCut(src: Mat, minArea: Double, maxArea: Double): MatOfPoint2f? {
        if (src.empty()) return null

        val bgr = Mat()
        when (src.channels()) {
            4 -> Imgproc.cvtColor(src, bgr, Imgproc.COLOR_RGBA2BGR)
            3 -> src.copyTo(bgr)
            else -> Imgproc.cvtColor(src, bgr, Imgproc.COLOR_GRAY2BGR)
        }

        val mask = Mat(src.rows(), src.cols(), CvType.CV_8UC1, Scalar(Imgproc.GC_BGD.toDouble()))
        val marginX = (src.cols() * 0.05).toInt().coerceAtLeast(2)
        val marginY = (src.rows() * 0.05).toInt().coerceAtLeast(2)
        val rect = Rect(
            marginX,
            marginY,
            (src.cols() - 2 * marginX).coerceAtLeast(2),
            (src.rows() - 2 * marginY).coerceAtLeast(2)
        )
        val bgd = Mat()
        val fgd = Mat()
        Imgproc.grabCut(bgr, mask, rect, bgd, fgd, 3, Imgproc.GC_INIT_WITH_RECT)

        val fgSure = Mat()
        val fgProb = Mat()
        Core.compare(mask, Scalar(Imgproc.GC_FGD.toDouble()), fgSure, Core.CMP_EQ)
        Core.compare(mask, Scalar(Imgproc.GC_PR_FGD.toDouble()), fgProb, Core.CMP_EQ)
        val fg = Mat()
        Core.bitwise_or(fgSure, fgProb, fg)

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        Imgproc.morphologyEx(fg, fg, Imgproc.MORPH_CLOSE, kernel)
        Imgproc.morphologyEx(fg, fg, Imgproc.MORPH_OPEN, kernel)

        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(fg, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        if (contours.isEmpty()) return null
        contours.sortByDescending { Imgproc.contourArea(it) }

        val bestQuad = findBestQuadFromContours(contours, minArea, maxArea, src.rows(), src.cols())
        if (bestQuad != null) return bestQuad

        for (c in contours) {
            val area = Imgproc.contourArea(c)
            if (area > maxArea) continue
            if (area < minArea) break
            val rr = Imgproc.minAreaRect(MatOfPoint2f(*c.toArray()))
            val points = arrayOf(Point(), Point(), Point(), Point())
            rr.points(points)
            return MatOfPoint2f(*points)
        }
        return null
    }

    private fun findBestDocumentRect(src: Mat, minArea: Double): Rect? {
        if (src.empty()) return null
        val gray = Mat()
        when (src.channels()) {
            4 -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
            3 -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
            else -> src.copyTo(gray)
        }
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(gray, gray)
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)

        val edges = Mat()
        Imgproc.Canny(gray, edges, 50.0, 150.0)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(7.0, 7.0))
        Imgproc.morphologyEx(edges, edges, Imgproc.MORPH_CLOSE, kernel)

        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(edges, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        if (contours.isEmpty()) return null
        contours.sortByDescending { Imgproc.contourArea(it) }

        var bestRect: Rect? = null
        var bestScore = Double.NEGATIVE_INFINITY
        for (c in contours) {
            val area = Imgproc.contourArea(c)
            if (area < minArea) break
            val r = Imgproc.boundingRect(c)
            val rectArea = (r.width * r.height).toDouble().coerceAtLeast(1.0)
            val fill = (area / rectArea).coerceIn(0.0, 1.0)
            val areaNorm = (rectArea / (src.rows().toDouble() * src.cols().toDouble())).coerceIn(0.0, 1.0)
            val score = areaNorm * 0.8 + fill * 0.2
            if (score > bestScore) {
                bestScore = score
                bestRect = expandRect(r, src.cols(), src.rows(), 0.03)
            }
        }
        return bestRect
    }

    private fun approxContourToQuad(c: MatOfPoint, epsFactor: Double): MatOfPoint2f? {
        val mat2f = MatOfPoint2f(*c.toArray())
        val peri = Imgproc.arcLength(mat2f, true)
        val approx = MatOfPoint2f()
        Imgproc.approxPolyDP(mat2f, approx, epsFactor * peri, true)
        return if (approx.total().toInt() == 4) approx else null
    }

    private fun distance(p1: Point, p2: Point): Double {
        return Math.hypot(p1.x - p2.x, p1.y - p2.y)
    }

    private fun scoreQuad(quad: MatOfPoint2f, rows: Int, cols: Int): Double {
        if (quad.total().toInt() != 4) return Double.NEGATIVE_INFINITY
        val pts = quad.toArray()
        val poly = MatOfPoint(*pts)
        if (!Imgproc.isContourConvex(poly)) return Double.NEGATIVE_INFINITY
        val area = abs(Imgproc.contourArea(poly))
        if (area <= 1.0) return Double.NEGATIVE_INFINITY

        val rect = Imgproc.boundingRect(poly)
        val rectArea = (rect.width * rect.height).toDouble().coerceAtLeast(1.0)
        val fill = (area / rectArea).coerceIn(0.0, 1.0)
        val areaNorm = (area / (rows.toDouble() * cols.toDouble())).coerceIn(0.0, 1.0)

        val cx = pts.map { it.x }.average()
        val cy = pts.map { it.y }.average()
        val dx = (cx - cols / 2.0) / cols
        val dy = (cy - rows / 2.0) / rows
        val centerScore = (1.0 - min(1.0, Math.hypot(dx, dy) * 2.0)).coerceIn(0.0, 1.0)
        val ordered = orderPoints(pts)
        val angleScore = rightAngleScore(ordered)
        val parallelScore = parallelOppositeScore(ordered)
        val w = (distance(ordered[0], ordered[1]) + distance(ordered[3], ordered[2])) / 2.0
        val h = (distance(ordered[0], ordered[3]) + distance(ordered[1], ordered[2])) / 2.0
        val ratio = if (h > 1.0) max(w, h) / min(w, h) else 99.0
        val aspectScore = scoreAspect(ratio)
        val minEdge = listOf(
            distance(ordered[0], ordered[1]),
            distance(ordered[1], ordered[2]),
            distance(ordered[2], ordered[3]),
            distance(ordered[3], ordered[0])
        ).minOrNull() ?: 0.0
        val minEdgeNorm = (minEdge / min(rows, cols).toDouble()).coerceIn(0.0, 1.0)
        val edgeSizeScore = (minEdgeNorm / 0.25).coerceIn(0.0, 1.0)
        val borderMarginScore = borderMarginScore(ordered, rows, cols)
        val framePenalty = imageFramePenalty(ordered, areaNorm, rows, cols)

        return areaNorm * 0.30 +
            fill * 0.16 +
            centerScore * 0.08 +
            angleScore * 0.16 +
            parallelScore * 0.12 +
            aspectScore * 0.05 +
            edgeSizeScore * 0.08 +
            borderMarginScore * 0.10 -
            framePenalty * 0.20
    }

    private fun refineQuadCorners(src: Mat, quad: MatOfPoint2f): MatOfPoint2f {
        if (quad.total().toInt() != 4 || src.empty()) return quad
        val ordered = orderPoints(quad.toArray())

        val gray = Mat()
        when (src.channels()) {
            4 -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
            3 -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
            else -> src.copyTo(gray)
        }
        Imgproc.GaussianBlur(gray, gray, Size(3.0, 3.0), 0.0)

        val corners = MatOfPoint()
        Imgproc.goodFeaturesToTrack(gray, corners, 120, 0.01, 10.0)
        if (corners.empty()) return MatOfPoint2f(*ordered)

        val radius = (min(src.cols(), src.rows()) * 0.12).coerceAtLeast(20.0)
        val refined = ordered.map { target ->
            var best = target
            var bestDist = Double.POSITIVE_INFINITY
            for (i in 0 until corners.rows()) {
                val xy = corners.get(i, 0) ?: continue
                if (xy.size < 2) continue
                val p = Point(xy[0], xy[1])
                val d = distance(target, p)
                if (d < radius && d < bestDist) {
                    bestDist = d
                    best = p
                }
            }
            best
        }.toTypedArray()
        return MatOfPoint2f(*refined)
    }

    private fun orderPoints(points: Array<Point>): Array<Point> {
        if (points.size != 4) return points

        val cx = points.map { it.x }.average()
        val cy = points.map { it.y }.average()

        val aroundCenter = points.sortedBy { Math.atan2(it.y - cy, it.x - cx) }.toMutableList()
        if (polygonSignedArea(aroundCenter) > 0) {
            aroundCenter.reverse()
        }

        val tlIndex = aroundCenter.indices.minByOrNull { i ->
            aroundCenter[i].x + aroundCenter[i].y
        } ?: 0
        val rotated = rotateLeft(aroundCenter, tlIndex)

        return if (rotated[1].y > rotated[3].y) {
            // Sequence is tl, bl, br, tr -> normalize ke tl, tr, br, bl
            arrayOf(rotated[0], rotated[3], rotated[2], rotated[1])
        } else {
            arrayOf(rotated[0], rotated[1], rotated[2], rotated[3])
        }
    }

    private fun polygonSignedArea(points: List<Point>): Double {
        var acc = 0.0
        for (i in points.indices) {
            val p = points[i]
            val q = points[(i + 1) % points.size]
            acc += (p.x * q.y) - (q.x * p.y)
        }
        return acc / 2.0
    }

    private fun rotateLeft(points: List<Point>, offset: Int): List<Point> {
        if (points.isEmpty()) return points
        val n = points.size
        val k = ((offset % n) + n) % n
        return (0 until n).map { i -> points[(i + k) % n] }
    }

    private fun rightAngleScore(ordered: Array<Point>): Double {
        if (ordered.size != 4) return 0.0
        var totalPenalty = 0.0
        for (i in 0 until 4) {
            val prev = ordered[(i + 3) % 4]
            val cur = ordered[i]
            val next = ordered[(i + 1) % 4]
            val a = Point(prev.x - cur.x, prev.y - cur.y)
            val b = Point(next.x - cur.x, next.y - cur.y)
            val denom = (Math.hypot(a.x, a.y) * Math.hypot(b.x, b.y)).coerceAtLeast(1e-6)
            val cosTheta = ((a.x * b.x + a.y * b.y) / denom).coerceIn(-1.0, 1.0)
            val theta = Math.toDegrees(acos(cosTheta))
            totalPenalty += abs(theta - 90.0)
        }
        val avgPenalty = totalPenalty / 4.0
        return (1.0 - (avgPenalty / 45.0)).coerceIn(0.0, 1.0)
    }

    private fun parallelOppositeScore(ordered: Array<Point>): Double {
        if (ordered.size != 4) return 0.0
        val a1 = normalizedSegmentAngle(ordered[0], ordered[1])
        val a2 = normalizedSegmentAngle(ordered[3], ordered[2])
        val b1 = normalizedSegmentAngle(ordered[1], ordered[2])
        val b2 = normalizedSegmentAngle(ordered[0], ordered[3])
        val d1 = minAngleDiff(a1, a2)
        val d2 = minAngleDiff(b1, b2)
        val avg = (d1 + d2) / 2.0
        return (1.0 - (avg / 30.0)).coerceIn(0.0, 1.0)
    }

    private fun normalizedSegmentAngle(p1: Point, p2: Point): Double {
        var angle = Math.toDegrees(Math.atan2(p2.y - p1.y, p2.x - p1.x))
        while (angle < 0.0) angle += 180.0
        while (angle >= 180.0) angle -= 180.0
        return angle
    }

    private fun minAngleDiff(a: Double, b: Double): Double {
        var diff = abs(a - b)
        if (diff > 90.0) diff = 180.0 - diff
        return abs(diff)
    }

    private fun borderMarginScore(ordered: Array<Point>, rows: Int, cols: Int): Double {
        if (ordered.size != 4) return 0.0
        val maxX = (cols - 1).toDouble().coerceAtLeast(1.0)
        val maxY = (rows - 1).toDouble().coerceAtLeast(1.0)
        val minMargins = ordered.map { p ->
            min(
                min(p.x, maxX - p.x),
                min(p.y, maxY - p.y)
            )
        }
        val avgMargin = (minMargins.average() / min(rows, cols).toDouble()).coerceIn(0.0, 1.0)
        return (avgMargin / 0.06).coerceIn(0.0, 1.0)
    }

    private fun imageFramePenalty(ordered: Array<Point>, areaNorm: Double, rows: Int, cols: Int): Double {
        if (ordered.size != 4) return 0.0
        val tolerance = min(rows, cols) * 0.02
        val maxX = (cols - 1).toDouble()
        val maxY = (rows - 1).toDouble()
        val corners = listOf(
            Point(0.0, 0.0),
            Point(maxX, 0.0),
            Point(maxX, maxY),
            Point(0.0, maxY)
        )
        var nearCount = 0
        for (i in 0 until 4) {
            if (distance(ordered[i], corners[i]) <= tolerance) nearCount++
        }
        if (nearCount >= 3 && areaNorm >= 0.90) return 1.0
        if (nearCount >= 2 && areaNorm >= 0.95) return 0.8
        return 0.0
    }

    private fun scoreAspect(ratio: Double): Double {
        // Dukung berbagai tipe dokumen: square/receipt, Letter, A-series, kartu ID.
        val targetRatios = listOf(
            1.0,               // hampir square (nota kecil)
            1.294,             // US Letter
            1.414,             // A-series (A4/A5)
            1.586,             // kartu ID/bank
            1.777              // dokumen panjang tertentu
        )
        var best = 0.0
        for (target in targetRatios) {
            val diff = abs(ratio - target)
            val s = (1.0 - min(1.0, diff / 0.9)).coerceIn(0.0, 1.0)
            if (s > best) best = s
        }
        return best
    }

    private fun deskewByHough(src: Mat): Mat {
        if (src.empty()) return src
        val gray = Mat()
        when (src.channels()) {
            4 -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
            3 -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
            else -> src.copyTo(gray)
        }
        Imgproc.GaussianBlur(gray, gray, Size(3.0, 3.0), 0.0)

        val edges = Mat()
        Imgproc.Canny(gray, edges, 75.0, 200.0)
        val lines = Mat()
        val minSide = min(src.cols(), src.rows()).toDouble()
        Imgproc.HoughLinesP(edges, lines, 1.0, Math.PI / 180.0, 60, minSide * 0.25, 20.0)
        if (lines.rows() == 0) return src

        val angles = mutableListOf<Double>()
        for (i in 0 until lines.rows()) {
            val l = lines.get(i, 0) ?: continue
            if (l.size < 4) continue
            val x1 = l[0]
            val y1 = l[1]
            val x2 = l[2]
            val y2 = l[3]
            val len = Math.hypot(x2 - x1, y2 - y1)
            if (len < minSide * 0.20) continue

            var angle = Math.toDegrees(Math.atan2(y2 - y1, x2 - x1))
            while (angle <= -90.0) angle += 180.0
            while (angle > 90.0) angle -= 180.0
            if (angle > 45.0) angle -= 90.0
            if (angle < -45.0) angle += 90.0
            if (abs(angle) <= 20.0) angles.add(angle)
        }
        if (angles.size < 5) return src

        angles.sort()
        val medianAngle = angles[angles.size / 2]
        if (abs(medianAngle) < 0.5) return src

        val center = Point(src.cols() / 2.0, src.rows() / 2.0)
        val rot = Imgproc.getRotationMatrix2D(center, -medianAngle, 1.0)
        val rotated = Mat()
        Imgproc.warpAffine(
            src,
            rotated,
            rot,
            Size(src.cols().toDouble(), src.rows().toDouble()),
            Imgproc.INTER_CUBIC,
            Core.BORDER_REPLICATE,
            Scalar(0.0)
        )
        return rotated
    }

    private fun cropToContent(src: Mat): Mat {
        if (src.empty()) return src
        val gray = Mat()
        when (src.channels()) {
            4 -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
            3 -> Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY)
            else -> src.copyTo(gray)
        }
        val mask = Mat()
        Imgproc.threshold(gray, mask, 8.0, 255.0, Imgproc.THRESH_BINARY)
        val nonZero = MatOfPoint()
        Core.findNonZero(mask, nonZero)
        if (nonZero.empty()) return src

        val rect = Imgproc.boundingRect(nonZero)
        if (rect.width < 10 || rect.height < 10) return src
        return Mat(src, rect).clone()
    }

    private fun expandRect(rect: Rect, maxW: Int, maxH: Int, ratio: Double): Rect {
        val dw = (rect.width * ratio).toInt().coerceAtLeast(1)
        val dh = (rect.height * ratio).toInt().coerceAtLeast(1)
        val expanded = Rect(rect.x - dw, rect.y - dh, rect.width + 2 * dw, rect.height + 2 * dh)
        return clampRectToMat(expanded, maxW, maxH)
    }

    private fun clampRectToMat(rect: Rect, maxW: Int, maxH: Int): Rect {
        val x = rect.x.coerceIn(0, maxW - 1)
        val y = rect.y.coerceIn(0, maxH - 1)
        val w = rect.width.coerceAtLeast(1).coerceAtMost(maxW - x)
        val h = rect.height.coerceAtLeast(1).coerceAtMost(maxH - y)
        return Rect(x, y, w, h)
    }

    private fun centerInsetRect(maxW: Int, maxH: Int, insetRatio: Double): Rect {
        val dx = (maxW * insetRatio).toInt().coerceAtLeast(1)
        val dy = (maxH * insetRatio).toInt().coerceAtLeast(1)
        val r = Rect(dx, dy, (maxW - 2 * dx).coerceAtLeast(1), (maxH - 2 * dy).coerceAtLeast(1))
        return clampRectToMat(r, maxW, maxH)
    }

    private fun saveCrop(mat: Mat): String {
        val safe = if (mat.empty()) Mat() else mat
        val toSave = if (safe.empty()) throw IllegalStateException("Crop result is empty") else safe
        val resultBitmap = Bitmap.createBitmap(toSave.cols(), toSave.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(toSave, resultBitmap)
        val outFile = File(cacheDir, "crop_${System.currentTimeMillis()}.png")
        outFile.outputStream().use {
            resultBitmap.compress(Bitmap.CompressFormat.PNG, 100, it)
        }
        return outFile.absolutePath
    }
}
