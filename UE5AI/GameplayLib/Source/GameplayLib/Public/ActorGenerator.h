// ActorGenerator.h
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "ActorGenerator.generated.h"

UENUM(BlueprintType, Blueprintable)
enum class EGeneratorState : uint8
{
    Idle,
    Active
};

UCLASS(Blueprintable, BlueprintType)
class GAMEPLAYLIB_API AActorGenerator : public AActor
{
    GENERATED_BODY()

public:
    AActorGenerator();

    virtual void Tick(float DeltaTime) override;
    virtual void BeginPlay() override;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Spawning")
    TSubclassOf<AActor> ActorToSpawn;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Spawning")
    float SpawnRadius;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Spawning")
    float SpawnInterval;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Spawning")
    int32 MaxSpawnedActors;


    UPROPERTY(BlueprintReadOnly, Category = "Spawning")
    int32 SpawnedActorsCount;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Spawning")
    EGeneratorState GeneratorState;

    UFUNCTION(BlueprintCallable, Category = "Spawning")
    void SetIdleState();

    UFUNCTION(BlueprintCallable, Category = "Spawning")
    void SetActiveState();
    
    
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Spawning", meta = (AllowPrivateAccess = "true"))
    class USphereComponent* SphereComponent;

	FTimerHandle SpawnTimer;
    
    
    UFUNCTION(BlueprintCallable, Category = "Spawning")
    void OnOverlapBegin(class UPrimitiveComponent* OverlappedComponent, class AActor* OtherActor, class UPrimitiveComponent* OtherComp, int32 OtherBodyIndex, bool bFromSweep, const FHitResult& SweepResult);

    UFUNCTION(BlueprintCallable, Category = "Spawning")
    void OnOverlapEnd(class UPrimitiveComponent* OverlappedComponent, class AActor* OtherActor, class UPrimitiveComponent* OtherComp, int32 OtherBodyIndex);
    

    UFUNCTION(BlueprintCallable, Category = "Spawning")
    void SpawnActor();

    UFUNCTION(BlueprintNativeEvent, Category = "Spawning")
    void OnActorSpawned(AActor* Actor);

    UFUNCTION(BlueprintNativeEvent, Category = "Spawning")
    bool CanTriggerSpawn(AActor* Actor);
};
