MITK_CREATE_MODULE( CESTIO
  DEPENDS MitkCEST MitkDICOM
  PACKAGE_DEPENDS PRIVATE nlohmann_json
  AUTOLOAD_WITH MitkDICOM
)
