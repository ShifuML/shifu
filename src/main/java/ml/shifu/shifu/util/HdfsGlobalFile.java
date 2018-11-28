package ml.shifu.shifu.util;

import ml.shifu.shifu.container.obj.RawSourceData;
import ml.shifu.shifu.fs.ShifuFileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.CompressionCodecFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.List;

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

public class HdfsGlobalFile {

    @SuppressWarnings("unused")
    private String filePath;
    private RawSourceData.SourceType sourceType;
    private List<FileStatus> fileStatsArr;

    private int currentFileOps;
    private BufferedReader reader;

    public HdfsGlobalFile(String filePath, RawSourceData.SourceType sourceType) throws IOException {
        this.filePath = filePath;
        this.sourceType = sourceType;

        this.fileStatsArr = ShifuFileUtils.getFileStatus(filePath, sourceType);
        this.currentFileOps = 0;
    }

    public String readLine() throws IOException {
        if (this.reader == null) {
            this.reader = createReaderFromPartFile(this.currentFileOps);
        }

        if (this.reader == null) {
            return null;
        }

        String line = this.reader.readLine();
        if (line == null) { // reach part file end
            // close current reader
            IOUtils.closeQuietly(this.reader);

            // reach part file end, open next part file
            this.currentFileOps++;
            this.reader = createReaderFromPartFile(this.currentFileOps);
            if (this.reader != null) {
                line = this.reader.readLine();
            }
        }

        return line;
    }

    private BufferedReader createReaderFromPartFile(int partIndex) throws IOException {
        if (this.fileStatsArr != null && partIndex < this.fileStatsArr.size()) {
            return new BufferedReader(
                    new InputStreamReader(openPartFileAsStream(this.fileStatsArr.get(partIndex))));
        } else {
            return null;
        }
    }

    private InputStream openPartFileAsStream(FileStatus fileStatus) throws IOException {
        CompressionCodecFactory compressionFactory = new CompressionCodecFactory(new Configuration());
        InputStream is = null;

        FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(sourceType);
        CompressionCodec codec = compressionFactory.getCodec(fileStatus.getPath());
        if (codec != null) {
            is = codec.createInputStream(fs.open(fileStatus.getPath()));
        } else {
            is = fs.open(fileStatus.getPath());
        }
        return is;
    }

    public void close() {
        IOUtils.closeQuietly(this.reader);
    }

    public void reset() {
        this.currentFileOps = 0;
        IOUtils.closeQuietly(this.reader);
        this.reader = null;
    }

}
