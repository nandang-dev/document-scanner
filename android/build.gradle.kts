allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory = rootProject.layout.buildDirectory.dir("../../build").get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

subprojects {
    plugins.withId("com.android.library") {
        if (name == "edge_detection") {
            val androidExt = extensions.findByName("android")
            if (androidExt != null) {
                val getNamespace = androidExt.javaClass.methods.firstOrNull { it.name == "getNamespace" }
                val setNamespace = androidExt.javaClass.methods.firstOrNull { it.name == "setNamespace" && it.parameterTypes.size == 1 }
                val currentNamespace = getNamespace?.invoke(androidExt) as? String
                if (currentNamespace.isNullOrBlank()) {
                    setNamespace?.invoke(androidExt, "com.sample.edgedetection")
                }
            }
        }
    }
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}

subprojects {
    if (name == "edge_detection") {
        tasks.configureEach {
            if (javaClass.name.contains("Kotlin", ignoreCase = true)) {
                val kotlinOptions = javaClass.methods
                    .firstOrNull { it.name == "getKotlinOptions" }
                    ?.invoke(this)
                kotlinOptions?.javaClass?.methods
                    ?.firstOrNull { it.name == "setJvmTarget" && it.parameterTypes.size == 1 }
                    ?.invoke(kotlinOptions, "1.8")
            }
        }

        tasks.withType(JavaCompile::class.java).configureEach {
            sourceCompatibility = "1.8"
            targetCompatibility = "1.8"
        }
    }
}

tasks.register<Exec>("startApi") {
    workingDir = rootProject.projectDir
    val os = org.gradle.internal.os.OperatingSystem.current()
    if (os.isWindows) {
        commandLine("powershell", "-Command", "Start-Process node -ArgumentList 'service' -WindowStyle Hidden")
    } else {
        commandLine("sh", "-c", "nohup node service >/dev/null 2>&1 &")
    }
}
