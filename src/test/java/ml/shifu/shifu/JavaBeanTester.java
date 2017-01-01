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
package ml.shifu.shifu;

import java.beans.IntrospectionException;
import java.beans.Introspector;
import java.beans.PropertyDescriptor;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Array;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.URL;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

public class JavaBeanTester {

    /**
     * Tests the get/set methods of the specified class.
     */
    public static <T> void test(final Class<T> clazz, boolean skipValidation, final String... skipThese)
            throws IntrospectionException {
        T bean = newInstance(clazz);
        if (bean == null) {
            throw new RuntimeException("Cannot create a bean class:" + clazz.getName());
        }
        final PropertyDescriptor[] props = Introspector.getBeanInfo(clazz).getPropertyDescriptors();
        List<String> skipList = Arrays.asList(skipThese);
        for (PropertyDescriptor prop : props) {
            // Check the list of properties that we don't want to test

            if (skipList.contains(prop.getName())) {
                continue;
            }
            final Method getter = prop.getReadMethod();
            final Method setter = findSetter(prop.getWriteMethod(), getter, clazz);

            Object actualValue = null;
            Object expectedValue = null;
            try {
                if (getter != null) {
                    actualValue = getter.invoke(bean);
                }

                if (setter != null) {
                    final Class<?>[] params = setter.getParameterTypes();

                    if (params.length == 1) {
                        // The set method has 1 argument, which is of the same type as the return type of the get
                        // method, so
                        // we can test this property
                        // Build a value of the correct type to be passed to the set method
                        Object value = buildValue(params[0]);

                        // Call the set method, then check the same value comes back out of the get method
                        setter.invoke(bean, value);

                        expectedValue = value;
                    }
                }
            } catch (IllegalArgumentException e) {
                throw new RuntimeException(e);
            } catch (SecurityException e) {
                throw new RuntimeException(e);
            } catch (InstantiationException e) {
                throw new RuntimeException(e);
            } catch (IllegalAccessException e) {
                throw new RuntimeException(e);
            } catch (InvocationTargetException e) {
                throw new RuntimeException(e);
            }

            if (getter != null && setter != null && !skipValidation) {
                if (!actualValue.equals(expectedValue)) {
                    throw new RuntimeException("After setter, getter gets wrong result: expectedValue:" + expectedValue
                            + " actualValue:" + actualValue);
                }
            }
        }

        // call three common method
        bean.toString();
        bean.equals(bean);
        bean.hashCode();
    }

    private static <T> Method findSetter(Method writeMethod, Method getter, Class<T> clazz) {
        if (writeMethod != null) {
            return writeMethod;
        }
        if (getter == null) {
            return null;
        }

        String setterName = getter.getName().replaceAll("get", "set");
        Method[] methods = clazz.getMethods();
        for (int i = 0; i < methods.length; i++) {
            if (setterName.equals(methods[i].getName())) {
                return methods[i];
            }
        }
        return null;
    }

    @SuppressWarnings("unchecked")
    private static <T> T newInstance(final Class<T> clazz) {
        Constructor<?>[] ctrs = clazz.getDeclaredConstructors();
        T initialInstance = null;
        Arrays.sort(ctrs, new Comparator<Constructor<?>>() {

            @Override
            public int compare(Constructor<?> o1, Constructor<?> o2) {
                int primitiveParamsSize1 = findPrimitiveParamsSize(o1);
                int primitiveParamsSize2 = findPrimitiveParamsSize(o2);
                return Integer.valueOf(primitiveParamsSize2).compareTo(primitiveParamsSize1);
            }

            private int findPrimitiveParamsSize(Constructor<?> o1) {
                int size = 0;
                for (int i = 0; i < o1.getParameterTypes().length; i++) {
                    if (o1.getParameterTypes()[i].isPrimitive() || o1.getParameterTypes()[i] == List.class
                            || o1.getParameterTypes()[i] == Map.class || o1.getParameterTypes()[i] == Set.class) {
                        size++;
                    }
                }
                return size;
            }

        });
        for (Constructor<?> ctr : ctrs) {
            try {
                ctr.setAccessible(true);
                if (ctr.getParameterTypes().length == 0) {
                    // The class has a no-arg constructor, so just call it
                    initialInstance = (T) ctr.newInstance();
                    return initialInstance;
                } else {
                    Object[] params = new Object[ctr.getParameterTypes().length];
                    for (int i = 0; i < ctr.getParameterTypes().length; i++) {
                        params[i] = buildValue(ctr.getParameterTypes()[i]);
                    }
                    return (T) ctr.newInstance(params);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        return null;
    }

    /**
     * Tests the get/set methods of the specified class.
     */
    public static <T> void test(final Class<T> clazz, final String... skipThese) throws IntrospectionException {
        test(clazz, true, skipThese);
    }

    @SuppressWarnings("rawtypes")
    private static Object buildValue(Class<?> clazz) throws InstantiationException, IllegalAccessException,
            IllegalArgumentException, SecurityException, InvocationTargetException {
        // Next check for a no-arg constructor
        // Specific rules for common classes
        if (clazz == String.class) {
            return "testvalue";

        } else if (clazz.isArray()) {
            return Array.newInstance(clazz.getComponentType(), 1);

        } else if (clazz == boolean.class || clazz == Boolean.class) {
            return true;

        } else if (clazz == int.class || clazz == Integer.class) {
            return 1;

        } else if (clazz == long.class || clazz == Long.class) {
            return 1L;

        } else if (clazz == double.class || clazz == Double.class) {
            return 1.0D;

        } else if (clazz == float.class || clazz == Float.class) {
            return 1.0F;

        } else if (clazz == char.class || clazz == Character.class) {
            return 'Y';
        } else if (clazz == List.class) {
            return new ArrayList();
        } else if (clazz == Map.class) {
            return new HashMap();
        } else if (clazz == Set.class) {
            return new HashSet();
        } else if (clazz.isEnum()) {
            return clazz.getEnumConstants()[0];
        }

        Class<?> newClazz = clazz;
        if (clazz.isInterface()) {
            throw new RuntimeException("Cannot find implement class for interface:" + clazz.getName());
//            List<Class<?>> allClassesByInterface = null;
//            try {
//                allClassesByInterface = getAllClassesByInterface(clazz, false);
//            } catch (Exception e) {
//                throw new RuntimeException("Cannot find implement class for interface:" + clazz.getName(), e);
//            }
//            if(allClassesByInterface == null || allClassesByInterface.size() == 0) {
//                throw new RuntimeException("Cannot find implement class for interface:" + clazz.getName());
//            }
//            newClazz = allClassesByInterface.get(0);
        }

        Object result = null;
        result = newInstance(newClazz);

        if (result != null) {
            return result;
        } else {
            throw new RuntimeException("Cannot instance parameter with class:" + newClazz.getName());
        }
    }

    public static List<Class<?>> getAllClassesByInterface(Class<?> interfaceClass, boolean samePackage)
            throws IOException, ClassNotFoundException, IllegalStateException {

        if (!interfaceClass.isInterface()) {
            throw new IllegalStateException("Class is not a interface.");
        }

        String packageName = samePackage ? interfaceClass.getPackage().getName() : "/";
        List<Class<?>> result = new ArrayList<Class<?>>();
        String[] classpaths = System.getProperty("java.class.path").split(";");
        for (int i = 0; i < classpaths.length; i++) {
            if (classpaths[i].endsWith("jar")) {
                if (classpaths[i].endsWith(".jar") || classpaths[i].endsWith(".zip")) {
                    ZipFile zip = new ZipFile(classpaths[i]);
                    Enumeration<? extends ZipEntry> entries = zip.entries();
                    while (entries.hasMoreElements()) {
                        ZipEntry entry = (ZipEntry) entries.nextElement();
                        String thisClassName = getClassName(entry);
                        if (thisClassName.endsWith(".class")) {
                            Class<?> tmpClazz = null;
                            try {
                                Thread.currentThread().getContextClassLoader().loadClass(thisClassName);
                                tmpClazz = Class.forName(thisClassName);
                            } catch (ClassNotFoundException e) {
                                e.printStackTrace();
                                continue;
                            }
                            if (interfaceClass.isAssignableFrom(tmpClazz) && !interfaceClass.equals(tmpClazz)) {
                                result.add(tmpClazz);
                            }
                        }
                    }
                    zip.close();
                }
            } else {
                result.addAll(findResources(interfaceClass, new File(classpaths[i]), packageName));
            }
        }
        return result;
    }

    private static String getClassName(ZipEntry entry) {
        StringBuffer className = new StringBuffer(entry.getName().replace('/', '.'));
        return className.toString();
    }

    public static List<Class<?>> findClasses(Class<?> interfaceClass, ClassLoader loader, String packageName)
            throws IOException, ClassNotFoundException {
        ClassLoader tmpLoader = loader;
        List<Class<?>> allClasses = new ArrayList<Class<?>>();
        // while(tmpLoader != null) {
        String packagePath = packageName.replace(".", "/");
        if (!packagePath.equals("/")) {
            Enumeration<URL> resources = tmpLoader.getResources(packagePath);
            while (resources.hasMoreElements()) {
                URL url = resources.nextElement();
                allClasses.addAll(findResources(interfaceClass, new File(url.getFile()), packageName));
            }
        } else {
            String path = tmpLoader.getResource("").getPath();
            allClasses.addAll(findResources(interfaceClass, new File(path), packageName));
        }
        // tmpLoader = loader.getParent();
        // }
        return allClasses;
    }

    @SuppressWarnings("unchecked")
    private static List<Class<?>> findResources(Class<?> interfaceClass, File directory, String packageName)
            throws ClassNotFoundException, IOException {

        List<Class<?>> results = new ArrayList<Class<?>>();
        if (directory == null || !directory.isDirectory())
            return Collections.EMPTY_LIST;
        
        File[] files = directory.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    if (!file.getName().contains(".")) {
                        if (!packageName.equals("/")) {
                            results.addAll(findResources(interfaceClass, file, packageName + "." + file.getName()));
                        } else {
                            results.addAll(findResources(interfaceClass, file, file.getName()));
                        }
                    }
                } else if (file.getName().endsWith(".class")) {
                    Class<?> clazz = null;
                    if (!packageName.equals("/")) {
                        clazz = Class.forName(packageName + "." + file.getName().substring(0, file.getName().length() - 6));
                    } else {
                        clazz = Class.forName(file.getName().substring(0, file.getName().length() - 6));
                    }
                    if (interfaceClass.isAssignableFrom(clazz) && !interfaceClass.equals(clazz)) {
                        results.add(clazz);
                    }
                }
            }
        } else {
            throw new IOException(String.format("Failed to list files in %s", directory.getAbsolutePath())); 
        }
        return results;
    }
}
