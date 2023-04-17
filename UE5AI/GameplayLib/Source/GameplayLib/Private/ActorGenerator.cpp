// ActorGenerator.cpp
#include "ActorGenerator.h"
#include "Components/SphereComponent.h"
#include "TimerManager.h"
#include "Kismet/GameplayStatics.h"
#include "GameFramework/Character.h"
#include "DrawDebugHelpers.h"


#if UE_BUILD_DEVELOPMENT
static TAutoConsoleVariable<int32> CVarShowSpawnRadius(
    TEXT("r.GameLib.ShowSpawnRadius"),
    0,
    TEXT("Show spawn radius for the actor generator.\n")
    TEXT("0: Disabled, 1: Enabled"),
    ECVF_Default);
#endif


AActorGenerator::AActorGenerator()
{
    PrimaryActorTick.bCanEverTick = true;

    SphereComponent = CreateDefaultSubobject<USphereComponent>(TEXT("SphereComponent"));
    RootComponent = SphereComponent;

    SpawnRadius = 500.0f;
    SpawnInterval = 1.0f;
    MaxSpawnedActors = 10;
    SpawnedActorsCount = 0;
    GeneratorState = EGeneratorState::Active;
    
    SphereComponent->SetSphereRadius(SpawnRadius);
}

void AActorGenerator::BeginPlay()
{
    Super::BeginPlay();

    if (HasAuthority())
    {
        SphereComponent->OnComponentBeginOverlap.AddDynamic(this, &AActorGenerator::OnOverlapBegin);
        SphereComponent->OnComponentEndOverlap.AddDynamic(this, &AActorGenerator::OnOverlapEnd);
    }
}

void AActorGenerator::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

#if UE_BUILD_DEVELOPMENT
    if (GetNetMode() != NM_DedicatedServer && CVarShowSpawnRadius.GetValueOnGameThread() != 0)
    {
        DrawDebugSphere(GetWorld(), GetActorLocation(), SpawnRadius, 24, FColor::Green, false, -1.f, 0, 1.f);
    }
#endif
}

void AActorGenerator::SpawnActor()
{
    if (HasAuthority() && GeneratorState == EGeneratorState::Active && SpawnedActorsCount < MaxSpawnedActors)
    {
        FVector RandomLocation = GetActorLocation() + FMath::VRand() * SpawnRadius;
        FVector StartTrace = RandomLocation + FVector(0, 0, 1000);
        FVector EndTrace = RandomLocation - FVector(0, 0, 1000);
        FHitResult HitResult;

        FCollisionQueryParams CollisionParams;
        CollisionParams.AddIgnoredActor(this);

        if (GetWorld()->LineTraceSingleByChannel(HitResult, StartTrace, EndTrace, ECC_Visibility, CollisionParams))
        {
            FVector SpawnLocation = HitResult.Location;
            FRotator SpawnRotation = FRotator::ZeroRotator;

            AActor* SpawnedActor = GetWorld()->SpawnActor<AActor>(ActorToSpawn, SpawnLocation, SpawnRotation);
            if (SpawnedActor)
            {
                SpawnedActorsCount++;
                OnActorSpawned(SpawnedActor);
                UE_LOG(LogTemp, Log, TEXT("Spawned Actor: %s. Total Spawned Actors: %d."), *SpawnedActor->GetName(), SpawnedActorsCount);
            }
        }
    }
}

void AActorGenerator::OnActorSpawned_Implementation(AActor* Actor)
{
}


bool AActorGenerator::CanTriggerSpawn_Implementation(AActor* Actor)
{
    return true;
}

void AActorGenerator::SetIdleState()
{
    if (HasAuthority())
    {
        GeneratorState = EGeneratorState::Idle;
        GetWorldTimerManager().ClearTimer(SpawnTimer);
        UE_LOG(LogTemp, Log, TEXT("Generator set to Idle state."));
    }
}

void AActorGenerator::SetActiveState()
{
    if (HasAuthority())
    {
        GeneratorState = EGeneratorState::Active;
        if (!GetWorldTimerManager().IsTimerActive(SpawnTimer))
        {
            GetWorldTimerManager().SetTimer(SpawnTimer, this, &AActorGenerator::SpawnActor, SpawnInterval, true);
        }
        UE_LOG(LogTemp, Log, TEXT("Generator set to Active state."));
    }
}

void AActorGenerator::OnOverlapBegin(class UPrimitiveComponent* OverlappedComponent, class AActor* OtherActor, class UPrimitiveComponent* OtherComp, int32 OtherBodyIndex, bool bFromSweep, const FHitResult& SweepResult)
{
	if (GeneratorState == EGeneratorState::Idle)
		return;

    if (OtherActor && CanTriggerSpawn(OtherActor))
    {
        GetWorldTimerManager().SetTimer(SpawnTimer, this, &AActorGenerator::SpawnActor, SpawnInterval, true);
    }
}



void AActorGenerator::OnOverlapEnd(class UPrimitiveComponent* OverlappedComponent, class AActor* OtherActor, class UPrimitiveComponent* OtherComp, int32 OtherBodyIndex)
{
    if (GeneratorState == EGeneratorState::Idle)
        return;
        
    if (OtherActor && CanTriggerSpawn(OtherActor))
    {
        GetWorldTimerManager().ClearTimer(SpawnTimer);
    }
}

