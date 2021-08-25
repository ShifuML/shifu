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
package ml.shifu.shifu.util;

import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;

/**
 * {@link HDFSUtils} is a unified class to get HDFS FileSystem Object.
 */
public final class HDFSUtils {

    private final static Logger LOG = LoggerFactory.getLogger(HDFSUtils.class);

    private final static String DEFAULT_HOST = "DEFAULT_HOST";

    /**
     * Conf object which is used to construct HDFS FileSystem.
     */
    private final static Configuration conf = new Configuration();

    /**
     * HDFS FileSystem
     *  Host -- FileSystem
     */
    private static volatile ConcurrentHashMap<String, FileSystem> hdfs = new ConcurrentHashMap<>();

    /**
     * Local FileSystem
     */
    private static volatile FileSystem lfs;

    private HDFSUtils() {
        // prevent new HDFSUtils();
    }

    public static Configuration getConf() {
        return conf;
    }
    
    public static boolean isDistributedMode(){
        String defaultFsName = conf.get("fs.defaultFS");
        return defaultFsName != null && defaultFsName.startsWith("hdfs:");
    }

    /**
     * Get HDFS FileSystem.
     * @return HDFS FileSystem handler
     */
    public static FileSystem getFS() {
        return getFS(null);
    }

    /**
     * Get HDFS FileSystem according the Path
     * @param path - file path to access
     * @return file system for specified path
     */
    public static FileSystem getFS(Path path) {
        String host = (path == null || path.toUri() == null || StringUtils.isBlank(path.toUri().getHost())) ?
                DEFAULT_HOST : path.toUri().getHost();
        FileSystem fs = hdfs.getOrDefault(host, null);
        if (fs == null) {
            try {
                if(path == null || StringUtils.isBlank(path.toUri().getScheme())) {
                    fs = FileSystem.get(conf);
                } else {
                    fs = path.getFileSystem(conf);
                }
                fs.setVerifyChecksum(false);
                hdfs.put(host, fs);
            } catch (IOException e) {
                LOG.error("Error on creating hdfs FileSystem object.", e);
                throw new ShifuException(ShifuErrorCode.ERROR_GET_HDFS_SYSTEM);
            }
        }
        return fs;
    }

    /*
     * Sometimes FileSystem will be close in NodeManger while no reason about that so far. Here we add a renew method to
     * create a new FileSystem instance. This should be package level but ShifuFileUtils is not in the same package.
     * 
     * @see ShifuFileUtils#getReader(String, ml.shifu.shifu.container.obj.RawSourceData.SourceType)
     */
    public static FileSystem renewFS(Path path) {
        String host = (path.toUri() == null || StringUtils.isBlank(path.toUri().getHost())) ?
                DEFAULT_HOST : path.toUri().getHost();
        hdfs.remove(host);
        return getFS(path);
    }

    /*
     * Get local FileSystem
     */
    public static FileSystem getLocalFS() {
        if(lfs == null) {
            synchronized(HDFSUtils.class) {
                if(lfs == null) {
                    try {
                        // initialization
                        lfs = FileSystem.getLocal(conf).getRaw();
                    } catch (IOException e) {
                        LOG.error("Error on creating local FileSystem object.", e);
                        throw new ShifuException(ShifuErrorCode.ERROR_GET_LOCAL_SYSTEM);
                    }
                }
            }
        }
        return lfs;
    }
    
    public static int getFileNumber(FileSystem fs, Path src) throws FileNotFoundException, IOException {
        RemoteIterator<LocatedFileStatus> itr = fs.listFiles(src, true);
        int total = 0;
        while(itr.hasNext()) {
            LocatedFileStatus cur = itr.next();
            String fileName = cur.getPath().getName();
            if (!fileName.startsWith("_") && !fileName.startsWith(".")) {
                total += 1;
            }
        }
        return total;
    }
}
