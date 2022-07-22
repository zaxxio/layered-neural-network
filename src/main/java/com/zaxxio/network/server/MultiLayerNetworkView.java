/*
 * MIT License
 *
 * Copyright (c) 2020 Partha Sutradhar.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.zaxxio.network.server;

import fi.iki.elonen.NanoHTTPD;

public class MultiLayerNetworkView extends NanoHTTPD {

    public static String DATA_NETWORK = "";

    public MultiLayerNetworkView(int port) {
        super(port);
    }

    @Override
    public Response serve(IHTTPSession session) {
        return newFixedLengthResponse(getText());
    }

    public String getText() {
        String msg = "<!DOCTYPE html>\n" +
                "<html lang=\"en\">\n" +
                "\n" +
                "<head>\n" +
                "    <meta charset=\"UTF-8\">\n" +
                "    <title>OpenBox</title>\n" +
                "</head>\n" +
                "<script type=\"application/javascript\" language=\"javascript\">\n" +
                "\n" +
                "    // Created by sinb on 16-12-23.\n" +
                "    // Modified from http://deepliquid.com/blog/archives/98\n" +
                "\n" +
                "    var arrow = [\n" +
                "        [2, 0],\n" +
                "        [-10, -4],\n" +
                "        [-10, 4]\n" +
                "    ];\n" +
                "    function drawFilledPolygon(ctx, shape) {\n" +
                "        ctx.beginPath();\n" +
                "        ctx.moveTo(shape[0][0], shape[0][1]);\n" +
                "\n" +
                "        for (p in shape)\n" +
                "            if (p > 0) ctx.lineTo(shape[p][0], shape[p][1]);\n" +
                "\n" +
                "        ctx.lineTo(shape[0][0], shape[0][1]);\n" +
                "        ctx.fillStyle = \"#34495e\";\n" +
                "        ctx.fill();\n" +
                "    }\n" +
                "    function translateShape(shape, x, y) {\n" +
                "        var rv = [];\n" +
                "        for (p in shape)\n" +
                "            rv.push([shape[p][0] + x, shape[p][1] + y]);\n" +
                "        return rv;\n" +
                "    }\n" +
                "    function rotateShape(shape, ang) {\n" +
                "        var rv = [];\n" +
                "        for (p in shape)\n" +
                "            rv.push(rotatePoint(ang, shape[p][0], shape[p][1]));\n" +
                "        return rv;\n" +
                "    }\n" +
                "    function rotatePoint(ang, x, y) {\n" +
                "        return [\n" +
                "            (x * Math.cos(ang)) - (y * Math.sin(ang)),\n" +
                "            (x * Math.sin(ang)) + (y * Math.cos(ang))\n" +
                "        ];\n" +
                "    }\n" +
                "    function drawLineArrow(ctx, x1, y1, x2, y2) {\n" +
                "        ctx.beginPath();\n" +
                "        ctx.moveTo(x1, y1);\n" +
                "        ctx.lineTo(x2, y2);\n" +
                "        ctx.strokeStyle = \"#34495e\";\n" +
                "        ctx.stroke();\n" +
                "        var ang = Math.atan2(y2 - y1, x2 - x1);\n" +
                "        drawFilledPolygon(ctx, translateShape(rotateShape(arrow, ang), x2, y2));\n" +
                "    };\n" +
                "\n" +
                "    window.onload = draw;\n" +
                "    function draw() {\n" +
                "        var canvas = document.getElementById(\"myCanvas\");\n" +
                "        var ctx = canvas.getContext(\"2d\");\n" +
                "        ctx.clearRect(0, 0, 1366, 500);\n" +
                "        var width = canvas.width;\n" +
                "        var height = canvas.height;\n" +
                "        canvas.style.backgroundColor = \"rgba(255, 255, 255, 1.0)\";\n" +
                "        ctx.font = \"13px Arial\";\n" +
                "\n" +
                "        var networkLayer = [" + MultiLayerNetworkView.DATA_NETWORK + "];\n" +
                "        var bias = false;\n" +
                "\n" +
                "        var maxNeuronNumInLayer = Math.max.apply(Math, networkLayer);\n" +
                "        var neuronSize = height / maxNeuronNumInLayer;\n" +
                "        var radius = getRadiusSize(neuronSize);\n" +
                "\n" +
                "        var intervalVertical = (height - maxNeuronNumInLayer * radius * 2) / maxNeuronNumInLayer;\n" +
                "        var interval = width / (networkLayer.length - 1) - radius;\n" +
                "\n" +
                "        var x = radius + 10;\n" +
                "        var y = 0;\n" +
                "        var neuronLocationPerLayer = [];\n" +
                "\n" +
                "        for (numberIdx in networkLayer) {\n" +
                "            var thisLayerNeuronLocation = [];\n" +
                "            number = networkLayer[numberIdx];\n" +
                "            console.log(\"x= \" + x);\n" +
                "            y = (height - number * neuronSize) / 2 + radius + intervalVertical;\n" +
                "\n" +
                "            for (var i = 0; i < number; ++i) {\n" +
                "                drawCircle(ctx, x, y, radius, '#3498db');\n" +
                "                if (bias === true) {\n" +
                "                    ctx.fillStyle = \"#34495e\";\n" +
                "                    ctx.textAlign = \"center\";\n" +
                "                    if (numberIdx != networkLayer.length - 1) {\n" +
                "                        if (i == 0)\n" +
                "                            ctx.fillText(\"+1\", x, y);\n" +
                "                    }\n" +
                "                }\n" +
                "\n" +
                "                thisLayerNeuronLocation.push([x, y]);\n" +
                "                y += (radius * 2 + intervalVertical);\n" +
                "            }\n" +
                "            neuronLocationPerLayer.push(thisLayerNeuronLocation);\n" +
                "            x += interval;\n" +
                "        }\n" +
                "        console.log(neuronLocationPerLayer);\n" +
                "        for (var i = 0; i < networkLayer.length - 1; i++) {\n" +
                "            var firstLayer = neuronLocationPerLayer[i];\n" +
                "            var secondLayer = neuronLocationPerLayer[i + 1];\n" +
                "            for (firstIdx in firstLayer) {\n" +
                "                var firstX = firstLayer[firstIdx][0];\n" +
                "                var firstY = firstLayer[firstIdx][1];\n" +
                "                for (secondIdx in secondLayer) {\n" +
                "                    var secondX = secondLayer[secondIdx][0];\n" +
                "                    var secondY = secondLayer[secondIdx][1];\n" +
                "                    if (bias === true) {\n" +
                "                        if (secondIdx == 0 && (i + 1) != networkLayer.length - 1) {\n" +
                "                            console.log(secondIdx)\n" +
                "                            continue;\n" +
                "                        }\n" +
                "                        else\n" +
                "                            drawLineArrow(ctx, firstX + radius, firstY, secondX - radius, secondY);\n" +
                "                    }\n" +
                "                    else {\n" +
                "                        drawLineArrow(ctx, firstX + radius, firstY, secondX - radius, secondY);\n" +
                "                    }\n" +
                "                }\n" +
                "            }\n" +
                "        }\n" +
                "\n" +
                "\n" +
                "    }\n" +
                "    function drawCircle(context, centerX, centerY, radius, color, txt) {\n" +
                "        context.beginPath();\n" +
                "        context.arc(centerX, centerY, radius, 0, 2 * Math.PI, false);\n" +
                "        context.fillStyle = color;\n" +
                "        context.fill();\n" +
                "        context.lineWidth = radius / 20;\n" +
                "        context.strokeStyle = '#34495e';\n" +
                "        context.stroke();\n" +
                "    }\n" +
                "\n" +
                "    function getRadiusSize(neuronSize) {\n" +
                "        return neuronSize / 2.2;\n" +
                "    }\n" +
                "\n" +
                "</script>\n" +
                "\n" +
                "<style>\n" +
                "    html,\n" +
                "    body {\n" +
                "        margin: 0;\n" +
                "        padding: 0;\n" +
                "        width: 100%;\n" +
                "        height: 100%;\n" +
                "        display: table\n" +
                "    }\n" +
                "\n" +
                "    #content {\n" +
                "        display: table-cell;\n" +
                "        text-align: center;\n" +
                "        vertical-align: middle;\n" +
                "    }\n" +
                "\n" +
                "    .center {\n" +
                "        text-align: center;\n" +
                "    }\n" +
                "</style>\n" +
                "\n" +
                "<body>\n" +
                "\n" +
                "    <div id=\"content\" class=\"center\">\n" + "<h1 style=\"text-align: center;text-transform: uppercase;color:#27ae60;\">Deep Neural Network</h1>\n" +
                "        <canvas id=\"myCanvas\" style='margin-left:30px;' width=\"1320\" height=\"520\"></canvas>\n" +
                "    </div>\n" +
                "\n" +
                "</body>\n" +
                "\n" +
                "</html>";
        return msg;
    }

}
