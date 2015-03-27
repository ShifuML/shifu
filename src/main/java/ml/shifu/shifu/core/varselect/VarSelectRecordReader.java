/*
 * Copyright [2013-2015] eBay Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.varselect;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.CompressionCodecFactory;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.util.LineReader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Copy from Hadoop LineRecordReder to support multiple file splits into one mapper task.
 */
public class VarSelectRecordReader extends RecordReader<LongWritable, Text> {
    private final static Logger LOG = LoggerFactory.getLogger(VarSelectRecordReader.class);

    private CompressionCodecFactory compressionCodecs = null;
    private long start;
    private long pos;
    private long end;
    private LineReader in;
    private int maxLineLength;
    private LongWritable key = null;
    private Text value = null;
    private byte[] recordDelimiterBytes;

    private TaskAttemptContext context;
    private FileSplit[] fileSplits;
    private int splitIndex = 0;
    private long wholeSize;
    private long comsumedSplitSize;

    public VarSelectRecordReader() {
    }

    public VarSelectRecordReader(byte[] recordDelimiter) {
        this.recordDelimiterBytes = recordDelimiter;
    }

    public void initialize(InputSplit genericSplit, TaskAttemptContext context) throws IOException {
        this.splitIndex = 0;
        this.context = context;
        this.fileSplits = ((VarSelectInputSplit) genericSplit).getFileSplits();
        initializeOne(context, this.fileSplits[this.splitIndex++]);
        for(FileSplit fileSplit: this.fileSplits) {
            this.wholeSize += fileSplit.getLength();
        }
    }

    private void initializeOne(TaskAttemptContext context, FileSplit split) throws IOException {
        Configuration job = context.getConfiguration();
        this.maxLineLength = job.getInt("mapred.linerecordreader.maxlength", Integer.MAX_VALUE);
        start = split.getStart();
        end = start + split.getLength();
        final Path file = split.getPath();
        compressionCodecs = new CompressionCodecFactory(job);
        final CompressionCodec codec = compressionCodecs.getCodec(file);

        // open the file and seek to the start of the split
        FileSystem fs = file.getFileSystem(job);
        FSDataInputStream fileIn = fs.open(split.getPath());
        boolean skipFirstLine = false;
        if(codec != null) {
            if(null == this.recordDelimiterBytes) {
                in = new LineReader(codec.createInputStream(fileIn), job);
            } else {
                in = new LineReader(codec.createInputStream(fileIn), job, this.recordDelimiterBytes);
            }
            end = Long.MAX_VALUE;
        } else {
            if(start != 0) {
                skipFirstLine = true;
                --start;
                fileIn.seek(start);
            }
            if(null == this.recordDelimiterBytes) {
                in = new LineReader(fileIn, job);
            } else {
                in = new LineReader(fileIn, job, this.recordDelimiterBytes);
            }
        }
        if(skipFirstLine) { // skip first line and re-establish "start".
            start += in.readLine(new Text(), 0, (int) Math.min((long) Integer.MAX_VALUE, end - start));
        }
        this.pos = start;
    }

    public boolean nextKeyValue() throws IOException {
        if(key == null) {
            key = new LongWritable();
        }
        key.set(pos);
        if(value == null) {
            value = new Text();
        }
        int newSize = 0;
        while(pos < end) {
            newSize = in.readLine(value, maxLineLength,
                    Math.max((int) Math.min(Integer.MAX_VALUE, end - pos), maxLineLength));
            if(newSize == 0) {
                break;
            }
            pos += newSize;
            if(newSize < maxLineLength) {
                break;
            }

            // line too long. try again
            LOG.info("Skipped line of size " + newSize + " at pos " + (pos - newSize));
        }
        if(newSize == 0) {
            if(this.splitIndex < this.fileSplits.length) {
                comsumedSplitSize += (end - start);
                // should close previous recorder here and the new one
                close();
                initializeOne(context, this.fileSplits[this.splitIndex++]);
                return true;
            } else {
                comsumedSplitSize += (end - start);
                key = null;
                value = null;
                return false;
            }
        } else {
            return true;
        }
    }

    @Override
    public LongWritable getCurrentKey() {
        return key;
    }

    @Override
    public Text getCurrentValue() {
        return value;
    }

    /**
     * Get the progress within the split TODO should be rewrite
     */
    public float getProgress() {
        if(start == end) {
            return 0.0f;
        } else {
            return Math.min(1.0f, (comsumedSplitSize + (pos - start)) / (float) (this.wholeSize));
        }
    }

    public synchronized void close() throws IOException {
        if(in != null) {
            in.close();
        }
    }
}
