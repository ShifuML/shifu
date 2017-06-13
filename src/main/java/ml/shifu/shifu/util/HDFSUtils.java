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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * {@link HDFSUtils} is a unified class to get HDFS FileSystem Object.
 */
public final class HDFSUtils {

    private final static Logger LOG = LoggerFactory.getLogger(HDFSUtils.class);

    /**
     * Conf object which is used to construct HDFS FileSystem.
     */
    private final static Configuration conf = new Configuration();

    /**
     * HDFS FileSystem
     */
    private static volatile FileSystem hdfs;

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

    /*
     * Get HDFS FileSystem
     */
    public static FileSystem getFS() {
        if(hdfs == null) {
            synchronized(HDFSUtils.class) {
                if(hdfs == null) {
                    try {
                        // initialization
                        // Assign to the hdfs instance after the tmpHdfs instance initialization fully complete.
                        // Avoid hdfs instance being used before fully initializaion.
                        FileSystem tmpHdfs = FileSystem.get(conf);
                        tmpHdfs.setVerifyChecksum(false);
                        hdfs = tmpHdfs;
                    } catch (IOException e) {
                        LOG.error("Error on creating hdfs FileSystem object.", e);
                        throw new ShifuException(ShifuErrorCode.ERROR_GET_HDFS_SYSTEM);
                    }
                }
            }
        }
        return hdfs;
    }

    /*
     * Sometimes FileSystem will be close in NodeManger while no reason about that so far. Here we add a renew method to
     * create a new FileSystem instance. This should be package level but ShifuFileUtils is not in the same package.
     * 
     * @see ShifuFileUtils#getReader(String, ml.shifu.shifu.container.obj.RawSourceData.SourceType)
     */
    public static FileSystem renewFS() {
        synchronized(HDFSUtils.class) {
            try {
                // initialization
                // Assign to the hdfs instance after the tmpHdfs instance initialization fully complete.
                // Avoid hdfs instance being used before fully initializaion.
                FileSystem tmpHdfs = FileSystem.get(conf);
                tmpHdfs.setVerifyChecksum(false);
                hdfs = tmpHdfs;
            } catch (IOException e) {
                LOG.error("Error on creating hdfs FileSystem object.", e);
                throw new ShifuException(ShifuErrorCode.ERROR_GET_HDFS_SYSTEM);
            }
        }
        return hdfs;
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
}
