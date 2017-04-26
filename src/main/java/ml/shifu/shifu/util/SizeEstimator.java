/*
 * Copyright [2012-2015] PayPal Software Foundation
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

import java.lang.management.ManagementFactory;
import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import javax.management.MBeanServer;

/**
 * Copied from Apache Spark project to estimate size of a Java instance.
 */
public final class SizeEstimator {

    // Sizes of primitive types
    private final static long BYTE_SIZE = 1;
    private final static long BOOLEAN_SIZE = 1;
    private final static long CHAR_SIZE = 2;
    private final static long SHORT_SIZE = 2;
    private final static long INT_SIZE = 4;
    private final static long LONG_SIZE = 8;
    private final static long FLOAT_SIZE = 4;
    private final static long DOUBLE_SIZE = 8;

    // Alignment boundary for objects
    private final static int ALIGN_SIZE = 8;

    // A cache of ClassInfo objects for each class
    private static Map<Class<?>, ClassInfo> classInfos = new ConcurrentHashMap<Class<?>, ClassInfo>();

    // Object and pointer sizes are arch dependent
    private static boolean is64bit = false;

    // Size of an object reference
    // Based on https://wikis.oracle.com/display/HotSpotInternals/CompressedOops
    private static boolean isCompressedOops = false;
    private static int pointerSize = 4;

    // Minimum size of a java.lang.Object
    private static int objectSize = 8;

    static {
        initialize();
    }

    // Sets object size, pointer size based on architecture and CompressedOops settings from the JVM.
    private static void initialize() {
        is64bit = System.getProperty("os.arch").contains("64");
        isCompressedOops = getIsCompressedOops();

        objectSize = !is64bit ? 8 : (!isCompressedOops ? 16 : 12);
        pointerSize = is64bit && !isCompressedOops ? 8 : 4;
        classInfos.clear();
    }

    private static boolean getIsCompressedOops() {
        // This is only used by tests to override the detection of compressed oops. The test
        // actually uses a system property instead of a SparkConf, so we'll stick with that.
        try {
            String hotSpotMBeanName = "com.sun.management:type=HotSpotDiagnostic";
            MBeanServer server = ManagementFactory.getPlatformMBeanServer();

            // NOTE: This should throw an exception in non-Sun JVMs
            Class<?> hotSpotMBeanClass = Class.forName("com.sun.management.HotSpotDiagnosticMXBean");
            Method getVMMethod = hotSpotMBeanClass.getDeclaredMethod("getVMOption", Class.forName("java.lang.String"));

            Object bean = ManagementFactory.newPlatformMXBeanProxy(server, hotSpotMBeanName, hotSpotMBeanClass);
            // TODO: We could use reflection on the VMOption returned ?
            return getVMMethod.invoke(bean, "UseCompressedOops").toString().contains("true");
        } catch (Throwable e) {
            return false;
        }
    }

    /*
     * Cached information about each class. We remember two things: the "shell size" of the class (size of all
     * non-static fields plus the java.lang.Object size), and any fields that are pointers to objects.
     */
    private static class ClassInfo {
        public ClassInfo(long shellSize, List<Field> pointerFields) {
            this.shellSize = shellSize;
            this.pointerFields = pointerFields;
        }

        private final long shellSize;
        private final List<Field> pointerFields;
    }

    /*
     * The state of an ongoing size estimation. Contains a stack of objects to visit as well as an IdentityHashMap of
     * visited objects, and provides utility methods for enqueueing new objects to visit.
     */
    private static class SearchState {
        public SearchState(IdentityHashMap<Object, Object> visited) {
            super();
            this.visited = visited;
        }

        private IdentityHashMap<Object, Object> visited;

        List<Object> stack = new ArrayList<Object>();

        private long size = 0L;

        public void enqueue(Object obj) {
            if(obj != null && !visited.containsKey(obj)) {
                visited.put(obj, null);
                stack.add(obj);
            }
        }

        public boolean isFinished() {
            return stack.isEmpty();
        }

        public Object dequeue() {
            Object elem = stack.get(stack.size() - 1);
            stack.remove(stack.size() - 1);
            return elem;
        }

        /**
         * @return the size
         */
        public long getSize() {
            return size;
        }

        /**
         * @param size
         *            the size to set
         */
        public void setSize(long size) {
            this.size = size;
        }
    }

    /*
     * Estimate size of one object instance.
     */
    public static long estimate(Object obj) {
        return estimate(obj, new IdentityHashMap<Object, Object>());
    }

    private static long estimate(Object obj, IdentityHashMap<Object, Object> visited) {
        SearchState state = new SearchState(visited);
        state.enqueue(obj);
        while(!state.isFinished()) {
            visitSingleObject(state.dequeue(), state);
        }
        return state.getSize();
    }

    private static void visitSingleObject(Object obj, SearchState state) {
        Class<?> cls = obj.getClass();
        if(cls.isArray()) {
            visitArray(obj, cls, state);
        } else if(obj instanceof ClassLoader || obj instanceof Class) {
            // Hadoop JobConfs created in the interpreter have a ClassLoader, which greatly confuses
            // the size estimator since it references the whole REPL. Do nothing in this case. In
            // general all ClassLoaders and Classes will be shared between objects anyway.
        } else {
            ClassInfo classInfo = getClassInfo(cls);
            state.setSize(state.getSize() + classInfo.shellSize);
            for(Field field: classInfo.pointerFields) {
                try {
                    state.enqueue(field.get(obj));
                } catch (IllegalArgumentException e) {
                    throw new RuntimeException(e);
                } catch (IllegalAccessException e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }

    // Estimate the size of arrays larger than ARRAY_SIZE_FOR_SAMPLING by sampling.
    private static final long ARRAY_SIZE_FOR_SAMPLING = 200;
    private static final long ARRAY_SAMPLE_SIZE = 100; // should be lower than ARRAY_SIZE_FOR_SAMPLING

    private static void visitArray(Object array, Class<?> cls, SearchState state) {
        int length = Array.getLength(array);
        Class<?> elementClass = cls.getComponentType();

        // Arrays have object header and length field which is an integer
        long arrSize = alignSize(objectSize + INT_SIZE);

        if(elementClass.isPrimitive()) {
            arrSize += alignSize(length * primitiveSize(elementClass));
            state.size += arrSize;
        } else {
            arrSize += alignSize(length * pointerSize);
            state.size += arrSize;

            if(length <= ARRAY_SIZE_FOR_SAMPLING) {
                for(int i = 0; i < length; i++) {
                    state.enqueue(Array.get(array, i));
                }
            } else {
                // Estimate the size of a large array by sampling elements without replacement.
                double size = 0.0;
                Random rand = new Random(42);
                Set<Integer> drawn = new HashSet<Integer>((int) ARRAY_SAMPLE_SIZE);
                for(int i = 0; i < ARRAY_SAMPLE_SIZE; i++) {

                    int index = 0;
                    do {
                        index = rand.nextInt(length);
                    } while(drawn.contains(index));
                    drawn.add(index);
                    Object elem = Array.get(array, index);
                    size += SizeEstimator.estimate(elem, state.visited);
                }
                state.size += Double.valueOf(((length / (ARRAY_SAMPLE_SIZE * 1.0)) * size)).longValue();
            }
        }
    }

    private static long primitiveSize(Class<?> cls) {
        if(cls == byte.class) {
            return BYTE_SIZE;
        } else if(cls == boolean.class) {
            return BOOLEAN_SIZE;
        } else if(cls == char.class) {
            return CHAR_SIZE;
        } else if(cls == short.class) {
            return SHORT_SIZE;
        } else if(cls == int.class) {
            return INT_SIZE;
        } else if(cls == long.class) {
            return LONG_SIZE;
        } else if(cls == float.class) {
            return FLOAT_SIZE;
        } else if(cls == double.class) {
            return DOUBLE_SIZE;
        } else {
            throw new IllegalArgumentException("Non-primitive class " + cls + " passed to primitiveSize()");
        }
    }

    public static void main(String[] args) {

    }

    /*
     * Get or compute the ClassInfo for a given class.
     */
    private static ClassInfo getClassInfo(Class<?> cls) {
        // Check whether we've already cached a ClassInfo for this class
        if(cls == Object.class) {
            ClassInfo info = new ClassInfo(8L, new ArrayList<Field>());
            classInfos.put(cls, info);
            return info;
        }
        ClassInfo info = classInfos.get(cls);
        if(info != null) {
            return info;
        }

        Class<?> superClass = cls.getSuperclass();
        ClassInfo parent = getClassInfo(superClass);
        long shellSize = parent.shellSize;
        List<Field> pointerFields = parent.pointerFields;

        for(Field field: cls.getDeclaredFields()) {
            if(!Modifier.isStatic(field.getModifiers())) {
                Class<?> fieldClass = field.getType();
                if(fieldClass.isPrimitive()) {
                    shellSize += primitiveSize(fieldClass);
                } else {
                    field.setAccessible(true); // Enable future get()'s on this field
                    shellSize += pointerSize;
                    pointerFields.add(0, field);
                }
            }
        }

        shellSize = alignSize(shellSize);

        // Create and cache a new ClassInfo
        ClassInfo newInfo = new ClassInfo(shellSize, pointerFields);
        classInfos.put(cls, newInfo);
        return newInfo;
    }

    private static long alignSize(long size) {
        long rem = size % ALIGN_SIZE;
        if(rem == 0) {
            return size;
        } else {
            return (size + ALIGN_SIZE - rem);
        }
    }
}
