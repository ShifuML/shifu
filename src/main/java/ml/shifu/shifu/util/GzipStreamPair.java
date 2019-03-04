package ml.shifu.shifu.util;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.InputStream;
import java.util.zip.GZIPInputStream;

/**
 * Copyright [2013-2018] PayPal Software Foundation
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **/

public class GzipStreamPair {

    private DataInputStream input;

    private boolean isGzip;

    public GzipStreamPair(DataInputStream input, boolean isGzip) {
        this.input = input;
        this.isGzip = isGzip;
    }

    /**
     * @return the input
     */
    public DataInputStream getInput() {
        return input;
    }

    /**
     * @param input the input to set
     */
    public void setInput(DataInputStream input) {
        this.input = input;
    }

    /**
     * @return the isGzip
     */
    public boolean isGzip() {
        return isGzip;
    }

    /**
     * @param isGzip the isGzip to set
     */
    public void setGzip(boolean isGzip) {
        this.isGzip = isGzip;
    }

    public static GzipStreamPair isGZipFormat(InputStream input) {
        DataInputStream dis = null;
        // check if gzip or not
        boolean isGZip = false;
        try {
            byte[] header = new byte[2];
            BufferedInputStream bis = new BufferedInputStream(input);
            bis.mark(2);
            int result = bis.read(header);
            bis.reset();
            int ss = (header[0] & 0xff) | ((header[1] & 0xff) << 8);
            if(result != -1 && ss == GZIPInputStream.GZIP_MAGIC) {
                dis = new DataInputStream(new GZIPInputStream(bis));
                isGZip = true;
            } else {
                dis = new DataInputStream(bis);
                isGZip = false;
            }
        } catch (java.io.IOException e) {
            dis = new DataInputStream(input);
            isGZip = false;
        }
        return new GzipStreamPair(dis, isGZip);
    }

}