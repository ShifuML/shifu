/*
 * Copyright [2012-2014] PayPal Software Foundation
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
package ml.shifu.shifu.fs;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.HDFSUtils;
import ml.shifu.shifu.util.HdfsPartFile;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.CompressionCodecFactory;
import org.apache.hadoop.io.compress.SnappyCodec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import parquet.hadoop.Footer;
import parquet.hadoop.ParquetFileReader;

import java.io.*;
import java.util.*;
import java.util.zip.GZIPInputStream;

/**
 * ShifuFileUtils class encapsulate the file system interface from other components.
 * It provides the functions that for all kinds of file operation.
 * <p>
 * Caller need to pass the file path and SourceType to do file operation
 */
public class ShifuFileUtils {

    private static final Logger log = LoggerFactory.getLogger(ShifuFileUtils.class);
    private static final String PARQUET_POSTFIX = ".parquet";


    // avoid user to create instance
    private ShifuFileUtils() {
    }

    /**
     * Create an empty file, if file doesn't exist
     * if the file already exists, this method won't do nothing just return false
     * 
     * @param path
     *            - file path to create
     * @param sourceType
     *            - where to create file
     * @return - true : create an file, or false
     * @throws IOException
     *             - if any I/O exception in processing
     */
    public static boolean createFileIfNotExists(String path, SourceType sourceType) throws IOException {
        Path filePath = new Path(path);
        return getFileSystemBySourceType(sourceType, filePath).createNewFile(filePath);
    }

    /**
     * Create Directory if directory doesn't exist
     * if the directory exists, this method will do nothing
     * 
     * @param sourceFile
     *            - source file
     * @return operation status
     * @throws IOException
     *             - if any I/O exception in processing
     */
    public static boolean createDirIfNotExists(SourceFile sourceFile) throws IOException {
        return createDirIfNotExists(sourceFile.getPath(), sourceFile.getSourceType());
    }

    /**
     * Create Directory if directory doesn't exist
     * if the directory exists, this method will do nothing
     * 
     * @param path
     *            - directory path
     * @param sourceType
     *            - local/hdfs
     * @return operation status
     * @throws IOException
     *             any io exception
     */
    public static boolean createDirIfNotExists(String path, SourceType sourceType) throws IOException {
        Path filePath = new Path(path);
        return getFileSystemBySourceType(sourceType, filePath).mkdirs(filePath);
    }

    /**
     * Get buffered writer with <code>{@link Constants#DEFAULT_CHARSET}</code> for source file
     * !!! Notice, if the file exists, it will be overwritten.
     * !!! Warning: writer instance should be closed by caller.
     * 
     * @param sourceFile
     *            - source file
     * @return buffered writer with <code>{@link Constants#DEFAULT_CHARSET}</code>
     * @throws IOException
     *             - if any I/O exception in processing
     */
    public static BufferedWriter getWriter(SourceFile sourceFile) throws IOException {
        return getWriter(sourceFile.getPath(), sourceFile.getSourceType());
    }

    /**
     * Get buffered writer with <code>{@link Constants#DEFAULT_CHARSET}</code> for specified file.
     * !!! Notice, if the file exists, it will be overwritten.
     * !!! Warning: writer instance should be closed by caller.
     * 
     * @param path
     *            - file path
     * @param sourceType
     *            - local/hdfs
     * @return buffered writer with <code>{@link Constants#DEFAULT_CHARSET}</code>
     * @throws IOException
     *             - if any I/O exception in processing
     */
    public static BufferedWriter getWriter(String path, SourceType sourceType) throws IOException {
        Path filePath = new Path(path);
        return new BufferedWriter(new OutputStreamWriter(getFileSystemBySourceType(sourceType, filePath)
                .create(filePath), Constants.DEFAULT_CHARSET));
    }

    /**
     * Get buffered reader with <code>{@link Constants#DEFAULT_CHARSET}</code> for specified file
     * <p>
     * !!! Warning: reader instance should be closed by caller.
     * 
     * @param sourceFile
     *            - source file with <code>{@link Constants#DEFAULT_CHARSET}</code>
     * @throws IOException
     *             - if any I/O exception in processing
     * @return buffered reader
     */
    public static BufferedReader getReader(SourceFile sourceFile) throws IOException {
        return getReader(sourceFile.getPath(), sourceFile.getSourceType());
    }

    /**
     * Get buffered reader with <code>{@link Constants#DEFAULT_CHARSET}</code> for specified file
     * <p>
     * !!! Warning: reader instance should be closed by caller.
     * 
     * @param path
     *            - file path
     * @param sourceType
     *            - local/hdfs
     * @throws IOException
     *             - if any I/O exception in processing
     * @return buffered reader with <code>{@link Constants#DEFAULT_CHARSET}</code>
     */
    public static BufferedReader getReader(String path, SourceType sourceType) throws IOException {
        try {
            Path filePath = new Path(path);
            return new BufferedReader(new InputStreamReader(getCompressInputStream(
                    getFileSystemBySourceType(sourceType, filePath).open(filePath), filePath),
                    Constants.DEFAULT_CHARSET));
        } catch (IOException e) {
            // To manual fix a issue that FileSystem is closed exceptionally. Here we renew a FileSystem object to make
            // sure all go through such issues.
            if(e.getMessage() != null) {
                if(e.getMessage().toLowerCase().indexOf("filesystem closed") >= 0) {
                    if(sourceType == SourceType.HDFS) {
                        Path filePath = new Path(path);
                        return new BufferedReader(new InputStreamReader(HDFSUtils.renewFS(filePath).open(filePath),
                                Constants.DEFAULT_CHARSET));
                    }
                }
            }
            throw e;
        }
    }

    /**
     * Get InputStream from (Path, SourceType)
     * @param path - file path
     * @param sourceType - file type
     * @return InputStream of file(Path, SourceType)
     * @throws IOException - if fail to open file
     */
    public static InputStream getInputStream(Path path, SourceType sourceType) throws IOException {
        try {
            return getCompressInputStream(getFileSystemBySourceType(sourceType, path).open(path), path);
        } catch (IOException e) {
            // To manual fix a issue that FileSystem is closed exceptionally. Here we renew a FileSystem object to make
            // sure all go through such issues.
            if(e.getMessage() != null) {
                if(e.getMessage().toLowerCase().indexOf("filesystem closed") >= 0) {
                    if(sourceType == SourceType.HDFS) {
                        return HDFSUtils.renewFS(path).open(path);
                    }
                }
            }
            throw e;
        }
    }

    private static InputStream getCompressInputStream(FSDataInputStream fdis, Path path) throws IOException {
        String name = path.getName();
        if(name.toLowerCase().endsWith(".gz")) {
            return new GZIPInputStream(fdis);
        } else if(name.toLowerCase().endsWith(".bz2")) {
            return new BZip2CompressorInputStream(fdis);
        } else if(name.toLowerCase().endsWith(".snappy")) {
            Configuration conf = new Configuration();
            CompressionCodecFactory ccf = new CompressionCodecFactory(conf);
            CompressionCodec codec = ccf.getCodecByClassName(SnappyCodec.class.getName());
            return codec.createInputStream(fdis);
        } else {
            return fdis;
        }
    }

    /**
     * Get the data scanners for a list specified paths
     * if the file is directory, get all scanner of normal sub-files
     * if the file is normal file, get its scanner
     * !!! Notice, all hidden files (file name start with ".") will be skipped
     * !!! Warning: scanner instances should be closed by caller.
     * 
     * @param paths
     *            - file paths to get the scanner
     * @param sourceType
     *            - local/hdfs
     * @return scanners for specified paths
     * @throws IOException
     *             - if any I/O exception in processing
     */
    public static List<Scanner> getDataScanners(List<String> paths, SourceType sourceType) throws IOException {
        if(paths == null || sourceType == null) {
            throw new IllegalArgumentException("paths should not be null, sourceType should not be null.");
        }
        List<Scanner> scanners = new ArrayList<Scanner>();
        for(String path: paths) {
            scanners.addAll(getDataScanners(path, sourceType));
        }
        return scanners;
    }

    public static List<Scanner> getDataScanners(String path, SourceType sourceType) throws IOException {
        return getDataScanners(path, sourceType, null);
    }

    /**
     * Get the data scanners for some specified path
     * if the file is directory, get all scanner of normal sub-files
     * if the file is normal file, get its scanner
     * !!! Notice, all hidden files (file name start with ".") will be skipped
     * !!! Warning: scanner instances should be closed by caller.
     * 
     * @param path
     *            - file path to get the scanner
     * @param sourceType
     *            - local/hdfs
     * @param pathFilter
     *            the path filter
     * @return scanners for specified path
     * @throws IOException
     *             - if any I/O exception in processing
     */
    @SuppressWarnings("deprecation")
    public static List<Scanner> getDataScanners(String path, SourceType sourceType, final PathFilter pathFilter)
            throws IOException {
        Path p = new Path(path);
        FileSystem fs = getFileSystemBySourceType(sourceType, p);

        FileStatus[] listStatus;
        if(fs.getFileStatus(p).isDir()) {
            // for folder we need filter pig header files
            listStatus = fs.listStatus(p, new PathFilter() {
                @Override
                public boolean accept(Path path) {
                    boolean hiddenOrSuccessFile = path.getName().startsWith(Constants.HIDDEN_FILES)
                            || path.getName().equalsIgnoreCase("_SUCCESS");
                    if(pathFilter != null) {
                        return !hiddenOrSuccessFile && pathFilter.accept(path);
                    } else {
                        return !hiddenOrSuccessFile;
                    }
                }
            });
        } else {
            listStatus = new FileStatus[] { fs.getFileStatus(p) };
        }

        if(listStatus.length > 1) {
            Arrays.sort(listStatus, new Comparator<FileStatus>() {

                @Override
                public int compare(FileStatus f1, FileStatus f2) {
                    return f1.getPath().getName().compareTo(f2.getPath().getName());
                }

            });
        }

        List<Scanner> scanners = new ArrayList<Scanner>();
        for(FileStatus f: listStatus) {
            String filename = f.getPath().getName();

            if(f.isDir()) {
                log.warn("Skip - {}, since it's direcory, please check your configuration.", filename);
                continue;
            }

            log.debug("Creating Scanner for file: {} ", filename);
            if(filename.endsWith(Constants.GZ_SUFFIX)) {
                scanners.add(new Scanner(new GZIPInputStream(fs.open(f.getPath())), Constants.DEFAULT_CHARSET));
            } else if(filename.endsWith(Constants.BZ2_SUFFIX)) {
                scanners.add(new Scanner(new BZip2CompressorInputStream(fs.open(f.getPath())),
                        Constants.DEFAULT_CHARSET));
            } else if(filename.endsWith(Constants.SHIFU_OUTPUT_PARQUET_FORMAT)) {
                scanners.add(new Scanner(new ParquetInputStream(f.getPath(), Constants.DEFAULT_CHARSET), Constants.DEFAULT_CHARSET));
            } else {
                scanners.add(new Scanner(new BufferedInputStream(fs.open(f.getPath())), Constants.DEFAULT_CHARSET));
            }
        }

        return scanners;
    }

    /**
     * Get the data scanners for some specified path
     * if the file is directory, get all scanner of normal sub-files
     * if the file is normal file, get its scanner
     * !!! Notice, all hidden files (file name start with ".") will be skipped
     * !!! Warning: scanner instances should be closed by caller.
     * 
     * @param sourceFile
     *            - source file
     * @return scanners for source file
     * @throws IOException
     *             - if any I/O exception in processing
     */
    public static List<Scanner> getDataScanners(SourceFile sourceFile) throws IOException {
        return getDataScanners(sourceFile.getPath(), sourceFile.getSourceType());
    }

    /**
     * Get the BufferedReader for dath with @path and @sourceType
     * @param path - source file
     * @param sourceType - local / hdfs
     * @return - the BufferedReader for that path
     * @throws IOException - if any I/O exception in processing
     */
    public static List<BufferedReader> getBufferedReaders(String path, SourceType sourceType) throws IOException {
        return getBufferedReaders(path, sourceType, null);
    }

    /**
     * Get the BufferedReader for dath with @path and @sourceType
     * @param path - source file
     * @param sourceType - local / hdfs
     * @param pathFilter - path filter to filter file under path
     * @return - the BufferedReader for that path
     * @throws IOException - if any I/O exception in processing
     */
    public static List<BufferedReader> getBufferedReaders(String path, SourceType sourceType, final PathFilter pathFilter)
            throws IOException {
        Path p = new Path(path);
        FileSystem fs = getFileSystemBySourceType(sourceType, p);

        FileStatus[] listStatus;
        if(fs.getFileStatus(p).isDir()) {
            // for folder we need filter pig header files
            listStatus = fs.listStatus(p, new PathFilter() {
                @Override
                public boolean accept(Path path) {
                    boolean hiddenOrSuccessFile = path.getName().startsWith(Constants.HIDDEN_FILES)
                            || path.getName().equalsIgnoreCase("_SUCCESS");
                    if(pathFilter != null) {
                        return !hiddenOrSuccessFile && pathFilter.accept(path);
                    } else {
                        return !hiddenOrSuccessFile;
                    }
                }
            });
        } else {
            listStatus = new FileStatus[] { fs.getFileStatus(p) };
        }

        if(listStatus.length > 1) {
            Arrays.sort(listStatus, new Comparator<FileStatus>() {

                @Override
                public int compare(FileStatus f1, FileStatus f2) {
                    return f1.getPath().getName().compareTo(f2.getPath().getName());
                }

            });
        }

        List<BufferedReader> readers = new ArrayList<BufferedReader>();
        for(FileStatus f: listStatus) {
            String filename = f.getPath().getName();

            if(f.isDir()) {
                log.warn("Skip - {}, since it's direcory, please check your configuration.", filename);
                continue;
            }

            log.debug("Creating Scanner for file: {} ", filename);
            if(filename.endsWith(Constants.GZ_SUFFIX)) {
                readers.add(new BufferedReader(new InputStreamReader(new GZIPInputStream(fs.open(f.getPath())),
                        Constants.DEFAULT_CHARSET)));
            } else if(filename.endsWith(Constants.BZ2_SUFFIX)) {
                readers.add(new BufferedReader(new InputStreamReader(new BZip2CompressorInputStream(fs.open(f.getPath())),
                        Constants.DEFAULT_CHARSET)));
            } else {
                readers.add(new BufferedReader(new InputStreamReader(new BufferedInputStream(fs.open(f.getPath())),
                        Constants.DEFAULT_CHARSET)));
            }
        }

        return readers;
    }

    /**
     * Copy src file to dst file in the same FileSystem. Such as copy local source to local destination,
     * copy hdfs source to hdfs dest.
     * 
     * @param srcPath
     *            - source file to copy
     * @param destPath
     *            - destination file
     * @param sourceType
     *            - local/hdfs
     * @throws IOException
     *             - if any I/O exception in processing
     */
    public static void copy(String srcPath, String destPath, SourceType sourceType) throws IOException {
        if(StringUtils.isEmpty(srcPath) || StringUtils.isEmpty(destPath) || sourceType == null) {
            throw new IllegalArgumentException(String.format(
                    "Null or empty parameters srcDataPath:%s, dstDataPath:%s, sourceType:%s", srcPath, destPath,
                    sourceType));
        }

        Path destFilePath = new Path(destPath);
        Path srcFilePath = new Path(srcPath);
        FileSystem fs = getFileSystemBySourceType(sourceType, destFilePath);
        // delete all files in dst firstly because of different folder if has dstDataPath
        if(!fs.delete(destFilePath, true)) {
            // ignore delete failed, it's ok.
        }

        FileUtil.copy(fs, srcFilePath, fs, destFilePath, false, new Configuration());
    }

    /**
     * Move src file to dst file in the same FileSystem.
     * 
     * @param srcPath
     *            - source file to copy
     * @param destPath
     *            - destination file
     * @param sourceType
     *            - local/hdfs
     * @throws IOException
     *             - if any I/O exception in processing
     */
    public static void moveTo(String srcPath, String destPath, SourceType sourceType) throws IOException {
        if(StringUtils.isEmpty(srcPath) || StringUtils.isEmpty(destPath) || sourceType == null) {
            throw new IllegalArgumentException(String.format(
                    "Null or empty parameters srcDataPath:%s, dstDataPath:%s, sourceType:%s", srcPath, destPath,
                    sourceType));
        }

        Path destFilePath = new Path(destPath);
        Path srcFilePath = new Path(srcPath);
        FileSystem fs = getFileSystemBySourceType(sourceType, destFilePath);
        if(!fs.exists(destFilePath)) {
            throw new RuntimeException(destPath + " does not exist.");
        }

        FileUtil.copy(fs, srcFilePath, fs, destFilePath, false, new Configuration());
    }

    /**
     * Check the path is directory or not, the SourceType is used to find the file system
     * 
     * @param sourceFile
     *            - source file
     * @return - true, if the file is directory; or false
     * @throws IOException
     *             - if any I/O exception in processing
     */
    public static boolean isDir(SourceFile sourceFile) throws IOException {
        return isDir(sourceFile.getPath(), sourceFile.getSourceType());
    }

    /**
     * Check the path is directory or not, the SourceType is used to find the file system
     * 
     * @param path
     *            - the path of source file
     * @param sourceType
     *            - SourceType to find file system
     * @return - true, if the file is directory; or false
     * @throws IOException
     *             - if any I/O exception in processing
     */
    @SuppressWarnings("deprecation")
    public static boolean isDir(String path, SourceType sourceType) throws IOException {
        Path filePath = new Path(path);
        FileSystem fs = getFileSystemBySourceType(sourceType, filePath);
        FileStatus status = fs.getFileStatus(filePath);
        return status.isDir();
    }

    /**
     * According to SourceType to check whether file exists.
     * 
     * @param sourceFile
     *            - source file
     * @return - true if file exists, or false
     * @throws IOException
     *             - if any I/O exception in processing
     */
    public static boolean isFileExists(SourceFile sourceFile) throws IOException {
        return isFileExists(sourceFile.getPath(), sourceFile.getSourceType());
    }

    /**
     * According to SourceType to check whether file exists.
     * 
     * @param path
     *            - path of source file
     * @param sourceType
     *            - local/hdfs
     * @return - true if file exists, or false
     * @throws IOException
     *             - if any I/O exception in processing
     */
    public static boolean isFileExists(String path, SourceType sourceType) throws IOException {
        return isFileExists(new Path(path), sourceType);
    }

    /**
     * Move src file to dst file in the same FileSystem. Such as move local source to local destination,
     * move hdfs source to hdfs dest.
     * 
     * @param srcPath
     *            - source file to move
     * @param destPath
     *            - destination file
     * @param sourceType
     *            - local/hdfs
     * @return true if moving successfully, or false
     * @throws IOException
     *             - if any I/O exception in processing
     */
    public static boolean move(String srcPath, String destPath, SourceType sourceType) throws IOException {
        if(StringUtils.isEmpty(srcPath) || StringUtils.isEmpty(destPath) || sourceType == null) {
            throw new IllegalArgumentException(String.format(
                    "Null or empty parameters srcDataPath:%s, dstDataPath:%s, sourceType:%s", srcPath, destPath,
                    sourceType));
        }

        Path destFilePath = new Path(destPath);
        Path srcFilePath = new Path(srcPath);
        FileSystem fs = getFileSystemBySourceType(sourceType, destFilePath);
        // delete all files in dst firstly because of different folder if has dstDataPath
        if(!fs.delete(destFilePath, true)) {
            // ignore delete failed, it's ok.
        }

        if(fs.exists(srcFilePath)) {
            // copy file only when source file exists.
            fs.rename(srcFilePath, destFilePath);
            return true;
        }

        return false;
    }

    /**
     * According to SourceType to check whether file exists.
     * 
     * @param path
     *            - @Path of source file
     * @param sourceType
     *            - local/hdfs
     * @return - true if file exists, or false
     * @throws IOException
     *             - if any I/O exception in processing
     */
    public static boolean isFileExists(Path path, SourceType sourceType) throws IOException {
        FileSystem fs = getFileSystemBySourceType(sourceType, path);
        FileStatus[] fileStatusArr = fs.globStatus(path);
        return !(fileStatusArr == null || fileStatusArr.length == 0);
    }

    /**
     * Delete the file or directory recursively.
     * 
     * @param sourceFile
     *            - source file to check
     * @return operation status
     * @throws IOException
     *             - if any I/O exception in processing
     */
    public static boolean deleteFile(SourceFile sourceFile) throws IOException {
        return deleteFile(sourceFile.getPath(), sourceFile.getSourceType());
    }

    /**
     * Delete the file or directory recursively.
     * 
     * @param path
     *            - file or directory
     * @param sourceType
     *            - file source [local/HDFS]
     * @return operation status
     * @throws IOException
     *             - if any I/O exception in processing
     */
    public static boolean deleteFile(String path, SourceType sourceType) throws IOException {
        Path filePath = new Path(path);
        FileSystem fs = getFileSystemBySourceType(sourceType, filePath);
        return fs.delete(filePath, true);
    }

    /**
     * Expand the file path, allowing user to use regex just like when using `hadoop fs`
     * According the rules in glob, "{2,3}", "*" will be allowed
     * 
     * @param rawPath
     *            - the raw file path that may contains regex
     * @param sourceType
     *            - file source [local/HDFS]
     * @return - the file path list after expansion
     * @throws IOException
     *             - if any I/O exception in processing
     */
    public static List<String> expandPath(String rawPath, SourceType sourceType) throws IOException {
        Path filePath = new Path(rawPath);
        FileSystem fs = getFileSystemBySourceType(sourceType, filePath);
        FileStatus[] fsArr = fs.globStatus(filePath);

        List<String> filePathList = new ArrayList<String>();
        if(fsArr != null) {
            for(FileStatus fileStatus: fsArr) {
                filePathList.add(fileStatus.getPath().toString());
            }
        }

        return filePathList;
    }

    /**
     * Get the FileSystem, according the source type
     * 
     * @param sourceType
     *          - which kind of file system
     * @param path
     *          - path that could specified which file system to use
     * @return - file system handler
     */
    public static FileSystem getFileSystemBySourceType(SourceType sourceType, Path path) {
        if(sourceType == null) {
            throw new IllegalArgumentException("sourceType should not be null.");
        }

        switch(sourceType) {
            case HDFS:
                return HDFSUtils.getFS(path);
            case LOCAL:
                return HDFSUtils.getLocalFS();
            default:
                throw new IllegalStateException(String.format("No such source type - %s.", sourceType));
        }
    }

    public static List<ColumnConfig> searchColumnConfig(EvalConfig config, List<ColumnConfig> configList)
            throws IOException {
        String path = config.getModelsPath();

        if(StringUtils.isNotEmpty(path)) {
            Path filePath = new Path(path);
            FileSystem fs = ShifuFileUtils.getFileSystemBySourceType(config.getDataSet().getSource(), filePath);

            while(path.indexOf("/") > 0) {
                path = path.substring(0, path.lastIndexOf("/"));
                Path columnConfigFile = new Path(path + "/ColumnConfig.json");
                if(fs.exists(columnConfigFile)) {
                    log.info("Using config file in this column config : {}", columnConfigFile.toString());
                    return CommonUtils.loadColumnConfigList(columnConfigFile.toString(), config.getDataSet()
                            .getSource());

                }
            }
        }

        return configList;
    }

    public static List<String> readFilePartsIntoList(String filePath, SourceType sourceType) throws IOException {
        List<String> lines = new ArrayList<String>();

        Path path = new Path(filePath);
        FileSystem fs = getFileSystemBySourceType(sourceType, path);
        FileStatus[] fileStatsArr = getFilePartStatus(filePath, sourceType);

        CompressionCodecFactory compressionFactory = new CompressionCodecFactory(new Configuration());
        for(FileStatus fileStatus: fileStatsArr) {
            InputStream is = null;
            CompressionCodec codec = compressionFactory.getCodec(fileStatus.getPath());
            if(codec != null) {
                is = codec.createInputStream(fs.open(fileStatus.getPath()));
            } else {
                is = fs.open(fileStatus.getPath());
            }

            lines.addAll(IOUtils.readLines(is));
            IOUtils.closeQuietly(is);
        }

        return lines;
    }

    public static FileStatus[] getFilePartStatus(String filePath, SourceType sourceType) throws IOException {
        return getFilePartStatus(filePath, sourceType, new PathFilter() { // accept all files by default
            @Override
            public boolean accept(Path path) {
                return true;
            }
        });
    }

    public static FileStatus[] getFilePartStatus(String filePath, SourceType sourceType, final String partFilePrefix)
            throws IOException {
        // FIXME, should only skip _SUCCESS, .pig_header such files, not start from part, some files may not
        // start from part.
        return getFilePartStatus(filePath, sourceType, path -> path.getName().startsWith(partFilePrefix));
    }

    public static FileStatus[] getFilePartStatus(String filePath, SourceType sourceType, PathFilter pathFilter)
            throws IOException {
        Path path = new Path(filePath);
        FileSystem fs = getFileSystemBySourceType(sourceType, path);

        FileStatus[] fileStatsArr;
        try {
            fileStatsArr = fs.listStatus(path, pathFilter);
        } catch (Exception e) {
            // read from glob if not found in listStatus, it usually be a regex path
            fileStatsArr = fs.globStatus(new Path(filePath), pathFilter);
        }

        if(fileStatsArr == null || fileStatsArr.length == 0) {
            // protected by reading glob status again
            fileStatsArr = fs.globStatus(new Path(filePath), pathFilter);
        }

        return fileStatsArr;
    }

    public static List<FileStatus> getFileStatus(String filePath, SourceType sourceType) throws IOException {
        List<FileStatus> fileStatusList = new ArrayList<FileStatus>();
        Path path = new Path(filePath);
        FileSystem fs = getFileSystemBySourceType(sourceType, path);
        FileStatus[] fileStatusArr = fs.globStatus(path);
        if ( fileStatusArr != null && fileStatusArr.length > 0 ) {
            for ( FileStatus fileStatus : fileStatusArr ) {
                fetchFileStatus(fileStatus.getPath(), sourceType, fileStatusList);
            }
        }
        return fileStatusList;
    }

    private static void fetchFileStatus(Path filePath, SourceType sourceType, List<FileStatus> fileStatusList)
            throws IOException {
        FileSystem fs = getFileSystemBySourceType(sourceType, filePath);
        try {
            FileStatus[] fileStatsArr = fs.listStatus(filePath);
            for(FileStatus fileStatus : fileStatsArr) {
                if(fileStatus.isDirectory()) {
                    fetchFileStatus(fileStatus.getPath(), sourceType, fileStatusList);
                } else {
                    if(!isHiddenFile(fileStatus.getPath().getName())) {
                        fileStatusList.add(fileStatus);
                    }
                }
            }
        } catch (IOException e) {
            log.error("Fail to fetch file status for - {}", filePath.toString());
            throw e;
        }
    }

    public static boolean isHiddenFile(String fileName) {
        return StringUtils.isBlank(fileName) || fileName.startsWith(".") || fileName.startsWith("_");
    }

    public static int getFilePartCount(String filePath, SourceType sourceType) throws IOException {
        FileStatus[] fileStatsArr = getFilePartStatus(filePath, sourceType);
        return fileStatsArr.length;
    }

    public static boolean isPartFileAllGzip(String filePath, SourceType sourceType) throws IOException {
        FileStatus[] fileStatsArr = getFilePartStatus(filePath, sourceType);

        boolean isGzip = true;
        for(FileStatus fileStatus: fileStatsArr) {
            if(!fileStatus.getPath().toString().endsWith("gz")) {
                isGzip = false;
            }
        }
        return isGzip;
    }

    public static long getFileOrDirectorySize(String filePath, SourceType sourceType) throws IOException {
        long size = 0;

        FileStatus[] fileStatsArr = getFilePartStatus(filePath, sourceType);
        for(FileStatus fileStats: fileStatsArr) {
            size += fileStats.getLen();
        }
        return size;
    }

    public static boolean isCompressedFileOrDirectory(String filePath, SourceType sourceType) throws IOException {
        boolean isCompressedFile = false;

        FileStatus[] fileStatsArr = getFilePartStatus(filePath, sourceType);
        for(FileStatus fileStatus: fileStatsArr) {
            if(fileStatus.getPath().toString().endsWith("gz") || fileStatus.getPath().toString().endsWith("bz2")) {
                isCompressedFile = true;
                break;
            }
        }
        return isCompressedFile;
    }

    public static void writeLines(@SuppressWarnings("rawtypes") Collection collection, String filePath,
            SourceType sourceType) throws IOException {
        BufferedWriter writer = getWriter(filePath, sourceType);
        try {
            for(Object object: collection) {
                if(object != null) {
                    writer.write(object.toString());
                    writer.newLine();
                }
            }
        } finally {
            IOUtils.closeQuietly(writer);
        }
    }

    public static void copyToLocal(SourceFile sourceFile, String localOutputPath) throws IOException {
        copyToLocal(sourceFile, Constants.HADOOP_PART_PREFIX, localOutputPath);
    }

    public static void copyToLocal(SourceFile sourceFile, String partFilePrefix, String localOutputPath)
            throws IOException {
        HdfsPartFile hdfsPartFile = new HdfsPartFile(sourceFile.getPath(), sourceFile.getSourceType(), partFilePrefix);
        BufferedWriter writer = new BufferedWriter(new FileWriter(localOutputPath));
        String line = null;
        try {
            while((line = hdfsPartFile.readLine()) != null) {
                writer.write(line);
                writer.newLine();
            }
        } catch (Exception e) {
            // ignore
        } finally {
            IOUtils.closeQuietly(writer);
            hdfsPartFile.close();
        }
    }

    public static void sortFile(String fileToSort, final String delimiter, final int fieldNum, final boolean isNumber)
            throws IOException {
        InputStream inputStream = new FileInputStream(new File(fileToSort));
        List<String> fileLines = IOUtils.readLines(inputStream);
        Collections.sort(fileLines, new Comparator<String>() {
            @Override
            public int compare(String from, String to) {
                String[] fromFields = CommonUtils.split(from, delimiter);
                String[] toFields = CommonUtils.split(to, delimiter);
                if (isNumber) {
                    Double fromVal = Double.parseDouble(fromFields[fieldNum]);
                    Double toVal = Double.parseDouble(toFields[fieldNum]);
                    return Double.compare(fromVal, toVal);
                } else {
                    return fromFields[fieldNum].compareTo(toFields[fieldNum]);
                }
            }
        });
        IOUtils.closeQuietly(inputStream);
        FileUtils.writeLines(new File(fileToSort), fileLines);
    }

    public static boolean isParquetFile(String dataSetRawPath, SourceType sourceType) throws IOException {
        boolean isParquet = false;
        if (dataSetRawPath.endsWith(PARQUET_POSTFIX)) {
            isParquet = true;
        } else {
            if (isDir(dataSetRawPath, sourceType)) {
                FileStatus[] fileStats = getFilePartStatus(dataSetRawPath,
                        sourceType, path -> path.getName().endsWith(PARQUET_POSTFIX));
                isParquet = (fileStats != null && fileStats.length > 0);
            }
        }
        return isParquet;
    }

    public static String[] readParquetFileHeader(String dataSetRawPath, SourceType source) throws IOException {
        Path path = new Path(dataSetRawPath);
        FileSystem fs = getFileSystemBySourceType(source, path);
        List<Footer> footers = ParquetFileReader.readFooters(HDFSUtils.getConf(),
                fs.getFileLinkStatus(path), true);
        List<String> fields = new ArrayList<>();
        if (CollectionUtils.isNotEmpty(footers)) {
            footers.get(0).getParquetMetadata().getFileMetaData()
                    .getSchema().getFields().stream().forEach(x -> fields.add(x.getName()));
        }
        return fields.toArray(new String[0]);
    }
}
