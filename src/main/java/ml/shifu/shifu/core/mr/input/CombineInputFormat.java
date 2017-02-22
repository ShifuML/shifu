/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.mr.input;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ml.shifu.guagua.mapreduce.GuaguaMapReduceConstants;

import org.apache.hadoop.fs.BlockLocation;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Copy from GuaguaInputFormat to support combining multiple small file splits together for one task.
 */
public class CombineInputFormat extends TextInputFormat {

    private static final Logger LOG = LoggerFactory.getLogger(CombineInputFormat.class);

    private static final String TEXTINPUTFORMAT_RECORD_DELIMITER = "textinputformat.record.delimiter";

    public static final String SHIFU_VS_SPLIT_MAX_COMBINED_SPLIT_SIZE = "shifu.vs.split.maxCombinedSplitSize";

    public static final String SHIFU_VS_SPLIT_COMBINABLE = "shifu.vs.split.combinable";

    /*
     * Splitter building logic including master setting, also includes combining input feature like Pig.
     */
    @Override
    public List<InputSplit> getSplits(JobContext job) throws IOException {
        List<InputSplit> newSplits = null;
        boolean combinable = job.getConfiguration().getBoolean(SHIFU_VS_SPLIT_COMBINABLE, false);
        if(combinable) {
            @SuppressWarnings("deprecation")
            // use this deprecation method to make it works on 0.20.2
            long blockSize = FileSystem.get(job.getConfiguration()).getDefaultBlockSize();
            long combineSize = job.getConfiguration().getLong(SHIFU_VS_SPLIT_MAX_COMBINED_SPLIT_SIZE, blockSize);
            if(combineSize == 0) {
                combineSize = blockSize;
            }
            job.getConfiguration().setLong(GuaguaMapReduceConstants.MAPRED_MIN_SPLIT_SIZE, 1l);
            job.getConfiguration().setLong(GuaguaMapReduceConstants.MAPRED_MAX_SPLIT_SIZE, combineSize);
            // in hadoop 2.0 such keys are changed
            job.getConfiguration().setLong("mapreduce.input.fileinputformat.split.minsize", 1);
            job.getConfiguration().setLong("mapreduce.input.fileinputformat.split.maxsize", combineSize);
            List<InputSplit> splits = super.getSplits(job);
            LOG.debug("combine size:{}, splits:{}", combineSize, splits);
            newSplits = getFinalCombineSplits(splits, combineSize);
        } else {
            newSplits = getCommonSplits(job);
        }
        LOG.info("Input size: {}", newSplits.size());
        return newSplits;
    }

    /*
     * Copy from pig implementation, need to check this code logic.
     */
    protected List<InputSplit> getFinalCombineSplits(List<InputSplit> newSplits, long combineSize) throws IOException {
        List<List<InputSplit>> combinePigSplits;
        try {
            combinePigSplits = getCombineSplits(newSplits, combineSize);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }
        newSplits = new ArrayList<InputSplit>();
        for(List<InputSplit> inputSplits: combinePigSplits) {
            FileSplit[] fss = new FileSplit[inputSplits.size()];
            for(int i = 0; i < inputSplits.size(); i++) {
                fss[i] = (FileSplit) (inputSplits.get(i));
            }
            newSplits.add(new CombineInputSplit(fss));
        }
        return newSplits;
    }

    /*
     * Generate the list of files and make them into FileSplits.
     */
    protected List<InputSplit> getCommonSplits(JobContext job) throws IOException {
        long minSize = Math.max(getFormatMinSplitSize(), getMinSplitSize(job));
        long maxSize = getMaxSplitSize(job);

        // generate splits
        List<InputSplit> splits = new ArrayList<InputSplit>();
        List<FileStatus> files = listStatus(job);
        for(FileStatus file: files) {
            Path path = file.getPath();
            if(isPigOrHadoopMetaFile(path)) {
                continue;
            }
            FileSystem fs = path.getFileSystem(job.getConfiguration());
            long length = file.getLen();
            BlockLocation[] blkLocations = fs.getFileBlockLocations(file, 0, length);
            if((length != 0) && isSplitable(job, path)) {
                long blockSize = file.getBlockSize();
                long splitSize = computeSplitSize(blockSize, minSize, maxSize);

                long bytesRemaining = length;
                // here double comparison can be directly used because of no precision requirement
                while(((double) bytesRemaining) / splitSize > 1.1d) {
                    int blkIndex = getBlockIndex(blkLocations, length - bytesRemaining);
                    splits.add(new CombineInputSplit(new FileSplit(path, length - bytesRemaining, splitSize,
                            blkLocations[blkIndex].getHosts())));
                    bytesRemaining -= splitSize;
                }

                if(bytesRemaining != 0) {
                    splits.add(new CombineInputSplit(new FileSplit(path, length - bytesRemaining, bytesRemaining,
                            blkLocations[blkLocations.length - 1].getHosts())));
                }
            } else if(length != 0) {
                splits.add(new CombineInputSplit(new FileSplit(path, 0, length, blkLocations[0].getHosts())));
            } else {
                // Create empty hosts array for zero length files
                splits.add(new CombineInputSplit(new FileSplit(path, 0, length, new String[0])));
            }
        }

        // Save the number of input files in the job-conf
        job.getConfiguration().setLong(GuaguaMapReduceConstants.NUM_INPUT_FILES, files.size());

        LOG.debug("Total # of splits: {}", splits.size());
        return splits;
    }

    private static final class ComparableSplit implements Comparable<ComparableSplit> {
        private InputSplit rawInputSplit;
        private Set<Node> nodes;
        // id used as a tie-breaker when two splits are of equal size.
        private long id;

        ComparableSplit(InputSplit split, long id) {
            rawInputSplit = split;
            nodes = new HashSet<Node>();
            this.id = id;
        }

        void add(Node node) {
            nodes.add(node);
        }

        void removeFromNodes() {
            for(Node node: nodes)
                node.remove(this);
        }

        public InputSplit getSplit() {
            return rawInputSplit;
        }

        @Override
        public boolean equals(Object other) {
            if(other == null || !(other instanceof ComparableSplit))
                return false;
            return (compareTo((ComparableSplit) other) == 0);
        }

        @Override
        public int hashCode() {
            return 41;
        }

        @Override
        public int compareTo(ComparableSplit other) {
            try {
                long cmp = rawInputSplit.getLength() - other.rawInputSplit.getLength();
                // in descending order
                return cmp == 0 ? (id == other.id ? 0 : id < other.id ? -1 : 1) : cmp < 0 ? 1 : -1;
            } catch (IOException e) {
                throw new RuntimeException(e);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
    }

    private static class DummySplit extends InputSplit {
        private long length;

        @Override
        public String[] getLocations() {
            return null;
        }

        @Override
        public long getLength() {
            return length;
        }

        public void setLength(long length) {
            this.length = length;
        }
    }

    private static class Node {
        private long length = 0;
        private List<ComparableSplit> splits;
        private boolean sorted;

        Node() throws IOException, InterruptedException {
            length = 0;
            splits = new ArrayList<ComparableSplit>();
            sorted = false;
        }

        void add(ComparableSplit split) throws IOException, InterruptedException {
            splits.add(split);
            length++;
        }

        void remove(ComparableSplit split) {
            if(!sorted)
                sort();
            int index = Collections.binarySearch(splits, split);
            if(index >= 0) {
                splits.remove(index);
                length--;
            }
        }

        void sort() {
            if(!sorted) {
                Collections.sort(splits);
                sorted = true;
            }
        }

        List<ComparableSplit> getSplits() {
            return splits;
        }

        @SuppressWarnings("unused")
        public long getLength() {
            return length;
        }
    }

    public static List<List<InputSplit>> getCombineSplits(List<InputSplit> oneInputSplits, long maxCombinedSplitSize)
            throws IOException, InterruptedException {
        List<Node> nodes = new ArrayList<Node>();
        Map<String, Node> nodeMap = new HashMap<String, Node>();
        List<List<InputSplit>> result = new ArrayList<List<InputSplit>>();
        List<Long> resultLengths = new ArrayList<Long>();
        long comparableSplitId = 0;

        int size = 0, nSplits = oneInputSplits.size();
        InputSplit lastSplit = null;
        int emptyCnt = 0;
        for(InputSplit split: oneInputSplits) {
            if(split.getLength() == 0) {
                emptyCnt++;
                continue;
            }
            if(split.getLength() >= maxCombinedSplitSize) {
                comparableSplitId++;
                List<InputSplit> combinedSplits = new ArrayList<InputSplit>();
                combinedSplits.add(split);
                result.add(combinedSplits);
                resultLengths.add(split.getLength());
            } else {
                ComparableSplit csplit = new ComparableSplit(split, comparableSplitId++);
                String[] locations = split.getLocations();
                // sort the locations to stabilize the number of maps: PIG-1757
                Arrays.sort(locations);
                HashSet<String> locationSeen = new HashSet<String>();
                for(String location: locations) {
                    if(!locationSeen.contains(location)) {
                        Node node = nodeMap.get(location);
                        if(node == null) {
                            node = new Node();
                            nodes.add(node);
                            nodeMap.put(location, node);
                        }
                        node.add(csplit);
                        csplit.add(node);
                        locationSeen.add(location);
                    }
                }
                lastSplit = split;
                size++;
            }
        }

        if(nSplits > 0 && emptyCnt == nSplits) {
            // if all splits are empty, add a single empty split as currently an empty directory is
            // not properly handled somewhere
            List<InputSplit> combinedSplits = new ArrayList<InputSplit>();
            combinedSplits.add(oneInputSplits.get(0));
            result.add(combinedSplits);
        } else if(size == 1) {
            List<InputSplit> combinedSplits = new ArrayList<InputSplit>();
            combinedSplits.add(lastSplit);
            result.add(combinedSplits);
        } else if(size > 1) {
            // combine small splits
            Collections.sort(nodes, nodeComparator);
            DummySplit dummy = new DummySplit();
            // dummy is used to search for next split of suitable size to be combined
            ComparableSplit dummyComparableSplit = new ComparableSplit(dummy, -1);
            for(Node node: nodes) {
                // sort the splits on this node in descending order
                node.sort();
                long totalSize = 0;
                List<ComparableSplit> splits = node.getSplits();
                int idx;
                int lenSplits;
                List<InputSplit> combinedSplits = new ArrayList<InputSplit>();
                List<ComparableSplit> combinedComparableSplits = new ArrayList<ComparableSplit>();
                while(!splits.isEmpty()) {
                    combinedSplits.add(splits.get(0).getSplit());
                    combinedComparableSplits.add(splits.get(0));
                    int startIdx = 1;
                    lenSplits = splits.size();
                    totalSize += splits.get(0).getSplit().getLength();
                    long spaceLeft = maxCombinedSplitSize - totalSize;
                    dummy.setLength(spaceLeft);
                    idx = Collections.binarySearch(node.getSplits().subList(startIdx, lenSplits), dummyComparableSplit);
                    idx = -idx - 1 + startIdx;
                    while(idx < lenSplits) {
                        long thisLen = splits.get(idx).getSplit().getLength();
                        combinedSplits.add(splits.get(idx).getSplit());
                        combinedComparableSplits.add(splits.get(idx));
                        totalSize += thisLen;
                        spaceLeft -= thisLen;
                        if(spaceLeft <= 0)
                            break;
                        // find next combinable chunk
                        startIdx = idx + 1;
                        if(startIdx >= lenSplits)
                            break;
                        dummy.setLength(spaceLeft);
                        idx = Collections.binarySearch(node.getSplits().subList(startIdx, lenSplits),
                                dummyComparableSplit);
                        idx = -idx - 1 + startIdx;
                    }
                    if(totalSize > maxCombinedSplitSize / 2) {
                        result.add(combinedSplits);
                        resultLengths.add(totalSize);
                        removeSplits(combinedComparableSplits);
                        totalSize = 0;
                        combinedSplits = new ArrayList<InputSplit>();
                        combinedComparableSplits.clear();
                        splits = node.getSplits();
                    } else {
                        if(combinedSplits.size() != lenSplits)
                            throw new AssertionError("Combined split logic error!");
                        break;
                    }
                }
            }
            // handle leftovers
            List<ComparableSplit> leftoverSplits = new ArrayList<ComparableSplit>();
            Set<InputSplit> seen = new HashSet<InputSplit>();
            for(Node node: nodes) {
                for(ComparableSplit split: node.getSplits()) {
                    if(!seen.contains(split.getSplit())) {
                        // remove duplicates. The set has to be on the raw input split not the
                        // comparable input split as the latter overrides the compareTo method
                        // so its equality semantics is changed and not we want here
                        seen.add(split.getSplit());
                        leftoverSplits.add(split);
                    }
                }
            }

            if(!leftoverSplits.isEmpty()) {
                long totalSize = 0;
                List<InputSplit> combinedSplits = new ArrayList<InputSplit>();
                List<ComparableSplit> combinedComparableSplits = new ArrayList<ComparableSplit>();

                int splitLen = leftoverSplits.size();
                for(int i = 0; i < splitLen; i++) {
                    ComparableSplit split = leftoverSplits.get(i);
                    long thisLen = split.getSplit().getLength();
                    if(totalSize + thisLen >= maxCombinedSplitSize) {
                        removeSplits(combinedComparableSplits);
                        result.add(combinedSplits);
                        resultLengths.add(totalSize);
                        combinedSplits = new ArrayList<InputSplit>();
                        combinedComparableSplits.clear();
                        totalSize = 0;
                    }
                    combinedSplits.add(split.getSplit());
                    combinedComparableSplits.add(split);
                    totalSize += split.getSplit().getLength();
                    if(i == splitLen - 1) {
                        // last piece: it could be very small, try to see it can be squeezed into any existing splits
                        for(int j = 0; j < result.size(); j++) {
                            if(resultLengths.get(j) + totalSize <= maxCombinedSplitSize) {
                                List<InputSplit> isList = result.get(j);
                                for(InputSplit csplit: combinedSplits) {
                                    isList.add(csplit);
                                }
                                removeSplits(combinedComparableSplits);
                                combinedSplits.clear();
                                break;
                            }
                        }
                        if(!combinedSplits.isEmpty()) {
                            // last piece can not be squeezed in, create a new combined split for them.
                            removeSplits(combinedComparableSplits);
                            result.add(combinedSplits);
                        }
                    }
                }
            }
        }
        LOG.info("Total input paths (combined) to process : {}", result.size());
        return result;
    }

    /*
     * The following codes are for split combination: see PIG-1518
     */
    private static Comparator<Node> nodeComparator = new Comparator<Node>() {
        @Override
        public int compare(Node o1, Node o2) {
            long cmp = o1.length - o2.length;
            return cmp == 0 ? 0 : cmp < 0 ? -1 : 1;
        }
    };

    private static void removeSplits(List<ComparableSplit> splits) {
        for(ComparableSplit split: splits)
            split.removeFromNodes();
    }

    public static final String PIG_SCHEMA = "pig_schema";

    public static final String PIG_HEADER = "pig_header";

    public static final String HADOOP_SUCCESS = "_SUCCESS";

    public static final String BZ2 = "bz2";

    /**
     * Whether it is not pig or hadoop meta output file.
     * 
     * @param path
     *            the checked path
     * @return if it is pig or hadoop meta file
     */
    protected boolean isPigOrHadoopMetaFile(Path path) {
        return path.toString().indexOf(HADOOP_SUCCESS) >= 0 || path.toString().indexOf(PIG_HEADER) >= 0
                || path.toString().indexOf(PIG_SCHEMA) >= 0;
    }

    @Override
    public RecordReader<LongWritable, Text> createRecordReader(InputSplit split, TaskAttemptContext context) {
        String delimiter = context.getConfiguration().get(TEXTINPUTFORMAT_RECORD_DELIMITER);
        byte[] recordDelimiterBytes = null;
        if(null != delimiter)
            recordDelimiterBytes = delimiter.getBytes();
        return new CombineRecordReader(recordDelimiterBytes);
    }

}
