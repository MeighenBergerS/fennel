/// \file DetectorConstruction.cc
/// \brief Implementation of the DetectorConstruction class

#include "DetectorConstruction.hh"

#include "G4RunManager.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4Cons.hh"
#include "G4Orb.hh"
#include "G4Sphere.hh"
#include "G4Trd.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorConstruction::DetectorConstruction()
: G4VUserDetectorConstruction(),
  fLogicTarget(NULL),
  fTargetPV(nullptr),
  fTargetMaterial(NULL),
  fCheckOverlaps(true)
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorConstruction::~DetectorConstruction()
{ }

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  // Define materials 
  DefineMaterials();
  
  // Define volumes
  return DefineVolumes();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void DetectorConstruction::DefineMaterials()
{ 
  // Lead material defined using NIST Manager
  G4NistManager* nistManager = G4NistManager::Instance();
  // Air defined using NIST Manager
  nistManager->FindOrBuildMaterial("G4_AIR");
  // Water defined using NIST Manager
  fTargetMaterial = nistManager->FindOrBuildMaterial("G4_WATER");
  
  // Print materials
  G4cout << *(G4Material::GetMaterialTable()) << G4endl;

}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume* DetectorConstruction::DefineVolumes()
{  
  // Target parameters
  //
  G4double env_sizeXY = 5 * m, env_sizeZ = 40 * m;
   
  //     
  // World
  //
  G4double world_sizeXY = 1.2 * env_sizeXY;
  G4double world_sizeZ  = 1.2 * env_sizeZ;
  G4Material* world_mat = G4Material::GetMaterial("G4_AIR");
  
  G4Box* solidWorld =    
    new G4Box(
      "World",
      0.5 * world_sizeXY,
      0.5 * world_sizeXY,
      0.5 * world_sizeZ);
      
  G4LogicalVolume* logicWorld =                         
    new G4LogicalVolume(solidWorld,
                        world_mat,
                        "World");
                                   
  G4VPhysicalVolume* physWorld = 
    new G4PVPlacement(0,
                      G4ThreeVector(),
                      logicWorld,
                      "World",
                      0,
                      false,
                      0, 
                      fCheckOverlaps);              
  //     
  // The target
  //  
  G4Box* Target =    
    new G4Box(
      "Target",
      0.5 * env_sizeXY,
      0.5 * env_sizeXY,
      0.5 * env_sizeZ);
      
  G4LogicalVolume* fLogicTarget =                         
    new G4LogicalVolume(Target,
                        fTargetMaterial,
                        "Target");
  fTargetPV =
    new G4PVPlacement(0,
                      G4ThreeVector(),
                      fLogicTarget,
                      "Target",
                      logicWorld,
                      false,
                      0,
                      fCheckOverlaps);
  //
  //always return the physical World
  //
  return physWorld;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void DetectorConstruction::SetTargetMaterial(G4String materialName)
{
  G4NistManager* nistManager = G4NistManager::Instance();

  G4Material* pttoMaterial = 
              nistManager->FindOrBuildMaterial(materialName);

  if (fTargetMaterial != pttoMaterial) {
     if ( pttoMaterial ) {
        fTargetMaterial = pttoMaterial;
        if (fLogicTarget) fLogicTarget->SetMaterial(fTargetMaterial);
        G4cout 
          << G4endl 
          << "----> The target is made of " << materialName << G4endl;
     } else {
        G4cout 
          << G4endl 
          << "-->  WARNING from SetTargetMaterial : "
          << materialName << " not found" << G4endl;
     }
  }
}
 
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
