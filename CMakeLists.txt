cmake_minimum_required(VERSION 2.8.9)

project(GBMWizard)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/GBMWizard")
set(EXTENSION_CATEGORY "Wizards")
set(EXTENSION_CONTRIBUTORS "Andrew Beers (Massachussetts General Hospital)")
set(EXTENSION_DESCRIPTION "A step-by-step workflow that uses deep learning, Docker, and radiomics to fully analyze a GBM case.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/QTIM-Lab/SlicerGBMWizard/master/GBMWizard/Resources/Icons/GBMWizard.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/QTIM-Lab/SlicerGBMWizard/master/GBMWizard/Resources/Icons/GBMWizard.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(GBMWizard)
## NEXT_MODULE

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_CPACK})
