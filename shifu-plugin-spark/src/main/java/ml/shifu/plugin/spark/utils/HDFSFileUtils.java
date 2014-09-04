/**
 * Copyright [2012-2014] eBay Software Foundation
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
package ml.shifu.plugin.spark.utils;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;

// TODO: Improve exception handling

/**
 * Contains utils for dealing with HDFS or local filesystems specific to the spark stats code.
 * Considers all paths without any scheme prefix to be local filesystem paths.
 */
public class HDFSFileUtils {

    private Configuration hdfsConf;

    /**
     * Reads the "core-site.xml" and "hdfs-site.xml" files in the provided path.
     * Forms a Configuration object corresponding to HDFS.
     * Throws IOException if configuration files not found.
     * 
     * @param hadoopConfPath Path to the hadoop configuration directory
     * 
     */
    public HDFSFileUtils(String hadoopConfPath) throws IOException {
        this.hdfsConf = new Configuration();
        Path coreSitePath = new Path(hadoopConfPath + "/" + "core-site.xml");
        Path hdfsSitePath = new Path(hadoopConfPath + "/" + "hdfs-site.xml");
        this.hdfsConf.addResource(coreSitePath);
        this.hdfsConf.addResource(hdfsSitePath);
        FileSystem hdfs = null;
        try {
            hdfs = FileSystem.get(this.hdfsConf);
        } catch (IOException e) {
            System.out
                    .println("ERROR: Could not create instance of filesystem");
            e.printStackTrace();
        }

        if (hdfs instanceof LocalFileSystem) {
            throw new IOException("ERROR: Could not create instance of hdfs FileSystem. Please check hadoop configuration files, got path " + hadoopConfPath);
        }

        if (hdfs != null)
            hdfs.close();
    }

    public HDFSFileUtils(URI HDFSUri) throws IOException {
        this.hdfsConf= new Configuration();
        hdfsConf.set("fs.default.name", HDFSUri.toString());
        FileSystem hdfs = null;
        try {
            hdfs = FileSystem.get(this.hdfsConf);
        } catch (IOException e) {
            System.out
                    .println("ERROR: Could not create instance of filesystem");
            e.printStackTrace();
        }

        if (hdfs instanceof LocalFileSystem) {
            throw new IOException("ERROR: Could not create instance of hdfs FileSystem. Please check HDFS URI, got " + HDFSUri.toString());
        }

        if (hdfs != null)
            hdfs.close();
        
    }

    /**
     * Deletes files from HDFS/ local filesystem.
     * Assumes a path to be HDFS if "file:" schema not present.
     * @param strPath Path to the file on HDFS. Assumed to be an HDFS path in absence of schema.
     * @return success has a value of true if deletion was successful 
     * @throws IOException 
     * @throws URISyntaxException 
     */
    public boolean delete(String strPath) throws IOException, URISyntaxException {
        strPath= fullDefaultLocal(strPath);
        Path p = new Path(strPath);
        FileSystem fs = null;
        try {
            fs = p.getFileSystem(this.hdfsConf);
        } catch (IOException e1) {
            System.out.println("Cannot obtain FileSystem for " + p.toString());
            e1.printStackTrace();
        }

        try {
            return fs.delete(p, true);
        } catch (IOException e) {
            System.out.println("Cannot delete file " + p.toString());
            e.printStackTrace();
        }
        try {
            if (fs != null)
                fs.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return true;
    }

    /**
     * Returns the URI for HDFS root directory based on the Configuration object for this instance.
     * @return HDFSUri A String which is the URI for HDFS root directory.
     * @throws IOException
     */
    public String getHDFSUri() throws IOException {
        FileSystem hdfs = FileSystem.get(this.hdfsConf);
        String Uri = hdfs.getUri().toString();
        hdfs.close();
        return Uri;
    }

    /**
     *  Uploads localPath to HDFSDir if localPath is on the local filesystem, and returns path of file on HDFS. Treats localPath as local if no scheme is specified.
     * @param localPath
     * @param HDFSDir
     * @return
     * @throws Exception
     */
    public String uploadToHDFSIfLocal(String localPath, String HDFSDir)
            throws Exception {
        localPath= fullDefaultLocal(localPath);
        HDFSDir= relativeToFullHDFSPath(HDFSDir);
        if (isHDFS(localPath)) 
            return localPath;
        String basename = new Path(localPath).getName().toString();
        Path HDFSPath = new Path(HDFSDir + "/" + basename);
        FileSystem hdfs = FileSystem.get(this.hdfsConf);
        hdfs.copyFromLocalFile(new Path(localPath), HDFSPath);
        hdfs.close();
        return relativeToFullHDFSPath(HDFSPath.toString());
    }
    

    /**
     * Returns the home directory for the HDFS filesystem.
     * @return homeDir Home directory URI string.
     * @throws IOException
     */
    public String getHDFSHomeDir() throws IOException {
        FileSystem hdfs = FileSystem.get(this.hdfsConf);
        String homeDir = hdfs.getHomeDirectory().toString();
        hdfs.close();
        return homeDir;
    }

    /**
     * Converts path to full URI. Assumes a path to be local if no schema present.
     * @param path The path to be converted to full URI.
     * @return fullPath The full URI in string format.
     * @throws IOException 
     * @throws URISyntaxException 
     */
    public String fullDefaultLocal(String path) throws IOException, URISyntaxException {
        if(path.startsWith("file:"))
            return path;
        else if(path.startsWith("hdfs:"))
            return relativeToFullHDFSPath(path);
        else    // convert local path to full path
            return relativeToFullLocalPath(path);
    }
    
    
    /**
     * Converts a relative HDFS path to it's full path.
     * @param relPath  The HDFS path to be converted to full URI.
     * @return fullPath The full URI for relPath in String format.
     * @throws IOException
     * @throws URISyntaxException 
     */
    public String relativeToFullHDFSPath(String relPath) throws IOException, URISyntaxException {
        // TODO: convert only "hdfs:" schema to full URI if absent
        if(relPath.startsWith("file:"))
            return relPath;
        else if (relPath.startsWith("hdfs:")) {
            // assume path is full as we cannot verify presence/ absence of URI authority and port
            return relPath;
            /*
            // make sure path contains authority, port etc.
            URI uri= new URI(relPath);
            URI hdfsURI= new URI(getHDFSUri());
            URI fullURI= new URI(hdfsURI.getScheme(), hdfsURI.getAuthority(), uri.getPath(), null, null);
            return fullURI.toString();
            */
        }
        if (relPath.startsWith("/")) {
            // relPath relative to root
            return this.getHDFSUri() + relPath;
        } else {
            // assume that path is relative to home
            // if path starts with ~/ remove that portion
            if(relPath.startsWith("~"))
                if(relPath.length() > 2)
                    relPath= relPath.substring(2);
                else
                    relPath= "";

            return this.getHDFSHomeDir() + "/" + relPath;
        }
    }

    
    /**
     * Converts a relative local path to it's full path.
     * @param relPath  The local path to be converted to full URI.
     * @return fullPath The full URI for relPath in String format.
     * @throws IOException
     * @throws URISyntaxException 
     */
    public String relativeToFullLocalPath(String path) throws IOException {
        if(path.startsWith("file:") || path.startsWith("hdfs:"))
            return path;
        if(path.startsWith("/"))
            return "file://" + path;
        else if(path.startsWith("~")) {
            // if path starts with ~/ remove that portion
            if(path.startsWith("~"))
                if(path.length() > 2)
                    path= path.substring(2);
                else
                    path= "";
            FileSystem localFS= FileSystem.get(new Configuration());
            Path homePath= localFS.getHomeDirectory();
            localFS.close();
            return homePath.toString() + "/" + path;            
        }
        else {   // path is relative to CWD
            // get CWD
            if(path.startsWith("."))
                // if path starts with ./ remove that portion
                if(path.length() > 2)
                    path= path.substring(2);
                else
                    path="";
            String workingDir = System.getProperty("user.dir");
            // add schema to local path
            return relativeToFullLocalPath(workingDir + "/" + path);
        }
    }


    /**
     * Concatenates all files in dirpath to target file.
     * @param target File to which files in dirpath should be concatenated to.
     * @param dirpath Source directory for files to be concatenated.
     * @throws IllegalArgumentException
     * @throws IOException
     */
    public void concat(String target, String dirpath)
            throws IllegalArgumentException, IOException {
        Path targetPath = new Path(target);
        FileSystem targetFS = targetPath.getFileSystem(this.hdfsConf);
        FileSystem hdfs = FileSystem.get(this.hdfsConf);
        targetFS.delete(targetPath, false);
        FileUtil.copyMerge(hdfs, new Path(dirpath), targetFS, new Path(target),
                true, this.hdfsConf, "");

        targetFS.close();
        hdfs.close();
    }
    
    /**
     * "touches" an empty file in local/ HDFS filesystem.
     * @param strPath File to be created. Default local.
     * @throws IOException
     * @throws URISyntaxException 
     */
    public void createEmptyFile(String strPath) throws IOException, URISyntaxException {
        strPath= fullDefaultLocal(strPath);
        Path path = new Path(strPath);
        FileSystem fs = path.getFileSystem(this.hdfsConf);
        fs.create(path);
    }
    
    /**
     * Returns Configuration object which contains HDFS configuration.
     * @return hdfsConf Configuration object
     */
    public Configuration getHDFSConf() {
        return this.hdfsConf;
    }
    
    /**
     * Returns boolean indicating whether a path is local or HDFS. This class will contain logic indicating whether 
     * paths without schema should be considered local or HDFS by default.
     * @param path
     * @return isLocal true if path is local
     */
    public boolean isLocal(String path) {
        return !isHDFS(path);
    }
    
    /**
     * Returns boolean indicating whether a path is local or HDFS. This class will contain logic indicating whether 
     * paths without schema should be considered local or HDFS by default.
     * @param path
     * @return isLocal true if path is HDFS
     */
    public boolean isHDFS(String path) {
        return path.startsWith("hdfs:");
    }
    
    /**
     * Checks if a path exists on HDFS/ local FS. Default local.
     * @param strPath
     * @return isExists boolean which is true if path exists.
     * @throws IllegalArgumentException
     * @throws IOException
     * @throws URISyntaxException
     */
    public boolean exists(String strPath) throws IllegalArgumentException, IOException, URISyntaxException {
        strPath= fullDefaultLocal(strPath);
        Path path= new Path(strPath);
        FileSystem fs= path.getFileSystem(this.hdfsConf);
        boolean isExists= fs.exists(path);
        fs.close();
        return isExists;        
    }
    
    /**
     * Copies file from/to local/HDFS filesystem paths. Both paths default to local.
     * @param source 
     * @param dest
     * @throws IOException
     * @throws URISyntaxException
     */
    
    public void copy(String source, String dest) throws IOException, URISyntaxException {
        source= fullDefaultLocal(source);
        dest= fullDefaultLocal(dest);
        FileSystem hdfs= FileSystem.get(hdfsConf);
        FileSystem local= FileSystem.get(new Configuration());
        
        if(isLocal(source) && isHDFS(dest)) {
            hdfs.copyFromLocalFile(new Path(source), new Path(dest));
        }
        else if(isHDFS(source) && isLocal(dest)) {
            hdfs.copyToLocalFile(new Path(source), new Path(dest));
        }
        else if(isHDFS(source) && isHDFS(dest)) {
            FileUtil.copy(hdfs, new Path(source), hdfs, new Path(dest), false, hdfsConf);
        }
        else {  // both paths local
            FileUtil.copy(local, new Path(source), local, new Path(dest), false, new Configuration());
        }            
    }
}
