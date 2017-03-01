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
package ml.shifu.shifu.guagua;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ml.shifu.guagua.hadoop.io.GuaguaInputSplit;
import ml.shifu.guagua.mapreduce.GuaguaInputFormat;
import ml.shifu.shifu.core.dtrain.CommonConstants;

import org.apache.commons.lang.ArrayUtils;
import org.apache.hadoop.fs.BlockLocation;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.InvalidInputException;
import org.apache.hadoop.mapreduce.security.TokenCache;
import org.apache.hadoop.util.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ShifuInputFormat extends GuaguaInputFormat {

    private static final Logger LOG = LoggerFactory.getLogger(ShifuInputFormat.class);

    private static final PathFilter hiddenFileFilter = new PathFilter() {
        public boolean accept(Path p) {
            String name = p.getName();
            return !name.startsWith("_") && !name.startsWith(".");
        }
    };

    private static class MultiPathFilter implements PathFilter {
        private List<PathFilter> filters;

        public MultiPathFilter(List<PathFilter> filters) {
            this.filters = filters;
        }

        public boolean accept(Path path) {
            for(PathFilter filter: filters) {
                if(!filter.accept(path)) {
                    return false;
                }
            }
            return true;
        }
    }

    /**
     * Splitter building logic including master setting, also includes combining input feature like Pig.
     */
    @Override
    public List<InputSplit> getSplits(JobContext job) throws IOException {
        List<InputSplit> newSplits = super.getSplits(job);
        String testDirs = job.getConfiguration().get("shifu.crossValidation.dir", "");
        LOG.info("Validation dir is {};", testDirs);
        if(org.apache.commons.lang.StringUtils.isNotBlank(testDirs)) {
            this.addCrossValidationDataset(newSplits, job);
        }
        return newSplits;
    }

    private FileSplit getFileSplit(FileSystem fs, FileStatus file, long offset, long length) throws IOException {
        BlockLocation[] blkLocations = fs.getFileBlockLocations(file, offset, length);
        List<String> hosts = new ArrayList<String>();
        for(BlockLocation location: blkLocations) {
            hosts.addAll(Arrays.asList(location.getHosts()));
        }
        String[] shosts = new String[hosts.size()];
        FileSplit fsp = new FileSplit(file.getPath(), offset, length, hosts.toArray(shosts));
        return fsp;
    }

    protected List<List<FileSplit>> getCrossValidationSplits(JobContext job, int count) throws IOException {
        LOG.debug("Split validation with count: {}", count);
        List<FileStatus> files = listCrossValidationStatus(job);
        List<FileSplit> current = new ArrayList<FileSplit>();
        List<List<FileSplit>> validationList = new ArrayList<List<FileSplit>>();
        long lengthSum = 0L;
        for(FileStatus file: files) {
            Path path = file.getPath();
            if(isPigOrHadoopMetaFile(path)) {
                continue;
            }
            lengthSum += file.getLen();
        }
        long size = lengthSum / count + 1;
        long remaining = 0L;
        for(FileStatus file: files) {
            Path path = file.getPath();
            if(isPigOrHadoopMetaFile(path)) {
                continue;
            }
            FileSystem fs = path.getFileSystem(job.getConfiguration());
            long offset = 0L;
            long length = file.getLen();
            if(length + remaining >= size) {
                long cut = (size - remaining) >= length ? length : (size - remaining);
                current.add(getFileSplit(fs, file, offset, cut));
                offset = cut;
                remaining = length - cut;
                validationList.add(current);
                current = new ArrayList<FileSplit>();
                while(remaining >= size) {
                    current.add(getFileSplit(fs, file, offset, size));
                    validationList.add(current);
                    current = new ArrayList<FileSplit>();
                    remaining -= size;
                    offset += size;
                }
                if(remaining > 0) {
                    current.add(getFileSplit(fs, file, offset, remaining));
                }

            } else {
                current.add(getFileSplit(fs, file, 0, length));
                remaining += length;
            }
        }
        if(current.size() > 0) {
            validationList.add(current);
        }

        LOG.debug("Total # of validationList: {}", validationList.size());
        return validationList;
    }

    protected void addCrossValidationDataset(List<InputSplit> trainingSplit, JobContext context) throws IOException {
        List<InputSplit> trainingNoMaster = new ArrayList<InputSplit>();
        for(InputSplit split: trainingSplit) {
            GuaguaInputSplit guaguaInput = (GuaguaInputSplit) split;
            if(guaguaInput.isMaster()) {
                continue;
            }
            trainingNoMaster.add(guaguaInput);
        }
        List<List<FileSplit>> csSplits = this.getCrossValidationSplits(context, trainingNoMaster.size());
        for(int i = 0; i < csSplits.size(); i++) {
            List<FileSplit> oneInput = csSplits.get(i);
            GuaguaInputSplit guaguaInput = (GuaguaInputSplit) trainingNoMaster.get(i);
            int trainingSize = guaguaInput.getFileSplits().length;
            FileSplit[] finalSplits = (FileSplit[]) ArrayUtils.addAll(guaguaInput.getFileSplits(),
                    oneInput.toArray(new FileSplit[0]));
            guaguaInput.setFileSplits(finalSplits);
            Boolean[] validationFlags = new Boolean[finalSplits.length];
            for(int j = 0; j < finalSplits.length; j++) {
                validationFlags[j] = j < trainingSize ? false : true;
            }
            guaguaInput.setExtensions(validationFlags);
        }
        LOG.info("Training input split size is: {}.", trainingSplit.size());
        LOG.info("Validation input split size is {}.", csSplits.size());
    }

    @SuppressWarnings("deprecation")
    protected List<FileStatus> listCrossValidationStatus(JobContext job) throws IOException {
        List<FileStatus> result = new ArrayList<FileStatus>();
        Path[] dirs = getInputPaths(job);
        if(dirs.length == 0) {
            throw new IOException("No input paths specified in job");
        }

        // get tokens for all the required FileSystems..
        TokenCache.obtainTokensForNamenodes(job.getCredentials(), dirs, job.getConfiguration());

        // Whether we need to recursive look into the directory structure
        boolean recursive = job.getConfiguration().getBoolean("mapreduce.input.fileinputformat.input.dir.recursive",
                false);

        List<IOException> errors = new ArrayList<IOException>();

        // creates a MultiPathFilter with the hiddenFileFilter and the
        // user provided one (if any).
        List<PathFilter> filters = new ArrayList<PathFilter>();
        filters.add(hiddenFileFilter);
        PathFilter jobFilter = getInputPathFilter(job);
        if(jobFilter != null) {
            filters.add(jobFilter);
        }
        PathFilter inputFilter = new MultiPathFilter(filters);

        for(int i = 0; i < dirs.length; ++i) {
            Path p = dirs[i];
            FileSystem fs = p.getFileSystem(job.getConfiguration());
            FileStatus[] matches = fs.globStatus(p, inputFilter);
            if(matches == null) {
                errors.add(new IOException("Input path does not exist: " + p));
            } else if(matches.length == 0) {
                errors.add(new IOException("Input Pattern " + p + " matches 0 files"));
            } else {
                for(FileStatus globStat: matches) {
                    if(globStat.isDir()) {
                        FileStatus[] fss = fs.listStatus(globStat.getPath());
                        for(FileStatus fileStatus: fss) {
                            if(inputFilter.accept(fileStatus.getPath())) {
                                if(recursive && fileStatus.isDir()) {
                                    addInputPathRecursive(result, fs, fileStatus.getPath(), inputFilter);
                                } else {
                                    result.add(fileStatus);
                                }
                            }
                        }
                    } else {
                        result.add(globStat);
                    }
                }
            }
        }

        if(!errors.isEmpty()) {
            throw new InvalidInputException(errors);
        }
        LOG.info("Total validation paths to process : " + result.size());
        return result;
    }

    @SuppressWarnings("deprecation")
    private void addInputPathRecursive(List<FileStatus> result, FileSystem fs, Path path, PathFilter inputFilter)
            throws IOException {
        FileStatus[] fss = fs.listStatus(path);
        for(FileStatus fileStatus: fss) {
            if(inputFilter.accept(fileStatus.getPath())) {
                if(fileStatus.isDir()) {
                    addInputPathRecursive(result, fs, fileStatus.getPath(), inputFilter);
                } else {
                    result.add(fileStatus);
                }
            }
        }
    }

    public static Path[] getInputPaths(JobContext context) {
        String dirs = context.getConfiguration().get(CommonConstants.CROSS_VALIDATION_DIR, "");
        LOG.info("crossValidation_dir:" + dirs);
        String[] list = StringUtils.split(dirs);
        Path[] result = new Path[list.length];
        for(int i = 0; i < list.length; i++) {
            result[i] = new Path(StringUtils.unEscapeString(list[i]));
        }
        return result;
    }
}
